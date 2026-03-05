from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import os
from dotenv import load_dotenv


load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(img: Image.Image) -> Image.Image:
    
   
    img = img.convert("RGB")

    
    img = ImageOps.exif_transpose(img)

    
    max_width = 1500
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)

   
    img = img.convert("L")

    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(2.0)

    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(1.1)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)

    img = img.point(lambda x: 0 if x < 140 else 255, '1')

    return img


@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()

        img = Image.open(io.BytesIO(image_bytes))

        img = preprocess_image(img)

        prompt = """
        Extract all handwritten text from this image clearly.
        Preserve line breaks and formatting exactly.
        Only return the text content.
        """

        response = model.generate_content([prompt, img])

        if not response.text:
            raise HTTPException(status_code=500, detail="Text extraction failed")

        return {
            "success": True,
            "extracted_text": response.text.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_image_from_path  # من الملف predict.py
import shutil
import os

app = FastAPI()

# إعداد CORS للسماح للفرونتند بالوصول إلى هذا السيرفر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # أو ضع رابط تطبيقك
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مجلد مؤقت لحفظ الصور
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # حفظ الصورة في مجلد مؤقت
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # استدعاء دالة التنبؤ من predict.py
    result = predict_image_from_path(file_location)

    # حذف الصورة بعد التنبؤ (اختياري)
    os.remove(file_location)

    return result

import uvicorn
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import keras
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import io
from PIL import Image

# تحديد مسار النموذج
#MODEL_PATH = os.getenv("MODEL_PATH", "model.h5")
# 1️⃣ تحميل النموذج
model = load_model("plant_disease_model_final.keras")

# 2️⃣ أسماء الفئات (38 class)
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# --------------------------
# 2️⃣ دالة لتحضير الصورة
# --------------------------
def prepare_image(image_bytes, img_size=(224,224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(img_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # مهم لـ EfficientNet
    return img
    
# 4️⃣ إعداد FastAPI
app = FastAPI(title="Plant Disease Diagnosis API")

# 5️⃣ تفعيل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# 4️⃣ Route للفحص
# --------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = prepare_image(image_bytes)

    predictions = model.predict(img)
    print("Output shape:", predictions.shape)
    class_idx = np.argmax(predictions, axis=1)[0]
    disease_id = class_names[class_idx] # مفتاح المرض 
    confidence = float(predictions[0][class_idx])

    return { "disease_id": disease_id, 
    "confidence": round(confidence * 100, 2) }

# --------------------------
# 5️⃣ تشغيل السيرفر
# --------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
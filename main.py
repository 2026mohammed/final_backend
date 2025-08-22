import uvicorn
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import io
from PIL import Image
app = FastAPI(title="Plant Disease Diagnosis API")
# تحديد مسار النموذج
#MODEL_PATH = os.getenv("MODEL_PATH", "model.h5")
# 1️⃣ تحميل النموذج
model = load_model("best_model (2).h5", compile=False)

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

# 3️⃣ قاموس العلاجات (احترافي)
treatments = {
    "Apple___Apple_scab": "استخدام مبيدات فطرية (مثل Captan أو Mancozeb)، تقليم الأشجار لزيادة التهوية.",
    "Apple___Black_rot": "إزالة الأفرع المصابة واستخدام مبيدات نحاسية دورياً.",
    "Apple___Cedar_apple_rust": "رش مبيدات وقائية (Myclobutanil)، التخلص من أشجار العرعر القريبة.",
    "Apple___healthy": "لا يوجد علاج، النبات سليم.",
    "Blueberry___healthy": "لا يوجد علاج، النبات سليم.",
    "Cherry_(including_sour)___Powdery_mildew": "استخدام الكبريت أو مبيدات الفطريات الحيوية، تحسين التهوية.",
    "Cherry_(including_sour)___healthy": "لا يوجد علاج، النبات سليم.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "استخدام مبيدات فطرية مثل Strobilurins، تناوب المحاصيل.",
    "Corn_(maize)___Common_rust_": "استخدام أصناف مقاومة، رش مبيدات فطرية إذا زادت الإصابة.",
    "Corn_(maize)___Northern_Leaf_Blight": "استخدام مبيدات فطرية (Azoxystrobin)، تحسين التباعد بين النباتات.",
    "Corn_(maize)___healthy": "لا يوجد علاج، النبات سليم.",
    "Grape___Black_rot": "رش مبيدات فطرية نحاسية، إزالة الأوراق المصابة.",
    "Grape___Esca_(Black_Measles)": "التقليم الجيد، إزالة الكروم المصابة بشدة.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "رش مبيدات فطرية نحاسية بشكل وقائي.",
    "Grape___healthy": "لا يوجد علاج، النبات سليم.",
    "Orange___Haunglongbing_(Citrus_greening)": "لا يوجد علاج مباشر، مكافحة الحشرة الناقلة (Psyllids)، استخدام شتلات سليمة.",
    "Peach___Bacterial_spot": "رش مبيدات نحاسية، استخدام أصناف مقاومة.",
    "Peach___healthy": "لا يوجد علاج، النبات سليم.",
    "Pepper,_bell___Bacterial_spot": "التخلص من النباتات المصابة، رش مبيدات نحاسية.",
    "Pepper,_bell___healthy": "لا يوجد علاج، النبات سليم.",
    "Potato___Early_blight": "استخدام مبيدات فطرية مثل Chlorothalonil أو Mancozeb.",
    "Potato___Late_blight": "الرش بمبيدات نحاسية، إزالة النباتات المصابة، تحسين الصرف.",
    "Potato___healthy": "لا يوجد علاج، النبات سليم.",
    "Raspberry___healthy": "لا يوجد علاج، النبات سليم.",
    "Soybean___healthy": "لا يوجد علاج، النبات سليم.",
    "Squash___Powdery_mildew": "استخدام الكبريت أو Potassium bicarbonate، تقليل الرطوبة.",
    "Strawberry___Leaf_scorch": "إزالة الأوراق المصابة، تحسين التهوية، استخدام مبيدات نحاسية.",
    "Strawberry___healthy": "لا يوجد علاج، النبات سليم.",
    "Tomato___Bacterial_spot": "استخدام مبيدات نحاسية، تجنب الري العلوي.",
    "Tomato___Early_blight": "استخدام Mancozeb أو Chlorothalonil، إزالة الأوراق السفلية المصابة.",
    "Tomato___Late_blight": "استخدام مبيدات نحاسية، تحسين التهوية، تجنب الري الزائد.",
    "Tomato___Leaf_Mold": "استخدام مبيدات فطرية وقائية، تقليم النباتات للتهوية.",
    "Tomato___Septoria_leaf_spot": "استخدام مبيدات فطرية (مثل Mancozeb)، إزالة الأوراق المصابة.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "استخدام زيت النيم أو مبيدات عناكب متخصصة.",
    "Tomato___Target_Spot": "استخدام مبيدات فطرية جهازية (مثل Azoxystrobin).",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "لا يوجد علاج مباشر، مكافحة الذبابة البيضاء.",
    "Tomato___Tomato_mosaic_virus": "إزالة النباتات المصابة، تعقيم الأدوات.",
    "Tomato___healthy": "لا يوجد علاج، النبات سليم."
}
# # ===== دالة تجهيز الصورة =====
def preprocess_image(image_bytes, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = preprocess_input(image_array)  # مهم: EfficientNet preprocessing
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# 5️⃣ تفعيل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== endpoint للتشخيص =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        prediction = model.predict(image_array)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))
        return JSONResponse(content={"class": predicted_class, "confidence": confidence})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ===== endpoint اختبار السيرفر =====
@app.get("/")
def read_root():
    return {"message": "API is running"}
import uvicorn
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import io
from PIL import Image

# تحديد مسار النموذج
MODEL_PATH = os.getenv("MODEL_PATH", "best_model (2).h5")
# 1️⃣ تحميل النموذج
#model = load_model("best_model (2).h5", compile=False)
model = load_model("best_model (2).h5", compile=True)

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
#@app.post("/predict")
#async def predict(file: UploadFile = File(...)):
 #   image_bytes = await file.read()
  #  img = prepare_image(image_bytes)

   # predictions = model.predict(img)
    #class_idx = np.argmax(predictions, axis=1)[0]
    #confidence = float(predictions[0][class_idx])

    #return {
     #   "disease": class_names[class_idx],
     #   "confidence": round(confidence * 100, 2),
      #  "treatment": treatments.get(class_names[class_idx], "لا توجد توصية علاجية متوفرة.")
    #}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = prepare_image(image_bytes)

    predictions = model.predict(img)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][class_idx])

    disease_id = class_names[class_idx]  # مفتاح المرض

    return {
        "disease_id": disease_id,
        "confidence": round(confidence * 100, 2)
    }


# --------------------------
# 5️⃣ تشغيل السيرفر
# --------------------------
#if __name__ == "__main__":
 #   uvicorn.run(app, host="0.0.0.0", port=8000)
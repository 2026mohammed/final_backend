from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input  # استخدم الشبكة المناسبة
import numpy as np
from PIL import Image
import io
import os
import uvicorn

# ==========================
# إعداد FastAPI
# ==========================
app = FastAPI(title="Model API")

# ==========================
# تحميل النموذج
# ==========================
model = load_model("best_model_render.h5", compile=False)
num_classes = model.output_shape[-1]

# ==========================
# دالة لمعالجة الصورة
# ==========================
def preprocess_image(file: bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # التطبيع حسب الشبكة
    return img_array

# ==========================
# نقطة النهاية للتصنيف
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents, target_size=(224, 224))
        
        # التنبؤ
        preds = model.predict(img_array)
        predicted_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})

# ==========================
# نقطة النهاية الرئيسية
# ==========================
@app.get("/")
def read_root():
    return {"message": "Model API is running!"}

# ==========================
# تشغيل التطبيق على Render
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render يوفر PORT تلقائيًا
    uvicorn.run(app, host="0.0.0.0", port=port)

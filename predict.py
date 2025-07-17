import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import requests
import tempfile

# Google Drive file ID
FILE_ID = "1AZuoRgRVVY1_OWJDlXr1IUtNT-xjyGLL"
# رابط التحميل المباشر للنموذج
url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# تحميل النموذج من Google Drive وحفظه مؤقتًا
response = requests.get(url)
response.raise_for_status()

with tempfile.NamedTemporaryFile(suffix=".h5") as temp_file:
    temp_file.write(response.content)
    temp_file.flush()

    # تحميل النموذج
    model = load_model(temp_file.name)

# الأصناف حسب ترتيب التدريب
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_YellowLeaf_Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]

# قاموس العلاجات
treatments = {
    'Pepper__bell___Bacterial_spot': "رش بمبيد نحاسي واستخدام بذور معتمدة.",
    'Pepper__bell___healthy': "النبات سليم، استمر في العناية الجيدة.",
    'Potato___Early_blight': "استخدم مبيدات فطرية مثل مانكوزيب وقلل الري الزائد.",
    'Potato___healthy': "البطاطس بصحة جيدة، لا حاجة لعلاج.",
    'Potato___Late_blight': "استخدم مبيدات مثل Metalaxyl وقلل الرطوبة.",
    'Tomato___Target_Spot': "إزالة الأوراق المصابة واستخدام مبيد فطري نحاسي.",
    'Tomato___Tomato_mosaic_virus': "إزالة النباتات المصابة وتعقيم الأدوات.",
    'Tomato___Tomato_YellowLeaf_Curl_Virus': "مكافحة الذباب الأبيض واستخدام نباتات مقاومة.",
    'Tomato_Bacterial_spot': "رش مبيدات نحاسية وتجنب الرش في المساء.",
    'Tomato_Early_blight': "رش بمبيد يحتوي على كلوروثالونيل أو مانكوزيب.",
    'Tomato_healthy': "النبات سليم، لا حاجة للعلاج.",
    'Tomato_Late_blight': "رش بمبيدات مثل Fosetyl-Al أو Chlorothalonil.",
    'Tomato_Leaf_Mold': "تحسين التهوية واستخدام مبيد فطري مناسب.",
    'Tomato_Septoria_leaf_spot': "إزالة الأوراق المصابة واستخدام مبيدات فطرية.",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "استخدم مبيد عناكب مثل Abamectin أو رش ماء بقوة."
}

def predict_image_from_path(file_path: str):
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    return predict_image(img_bytes)

def predict_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0]))

    treatment = treatments.get(predicted_class, "لا توجد توصية علاجية متاحة.")

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 3),
        "treatment": treatment
    }

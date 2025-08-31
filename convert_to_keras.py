import tensorflow as tf
import keras

# 1) المسار المحلي لمجلد النموذج
saved_model_dir = "finetuned-model"

# 2) تحميل النموذج كـ TFSMLayer
layer = keras.layers.TFSMLayer(
    saved_model_dir, 
    call_endpoint="serving_default"
)

# 3) إنشاء مدخل Keras Input مطابق لتوقيع النموذج
inp = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32, name="input_layer_1")

# 4) تمرير الإدخال عبر TFSMLayer
out = layer(inputs=inp)["output_0"]

# 5) لفه داخل نموذج Keras
model = keras.Model(inp, out)

# 6) حفظ النموذج كـ .keras
model.save("plant_disease_model_final.keras", save_format="keras_v3")
print("✅ تم حفظ النموذج محليًا بصيغة .keras")

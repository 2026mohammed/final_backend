import tensorflow as tf

model = tf.saved_model.load("finetuned-model")
print(list(model.signatures.keys()))

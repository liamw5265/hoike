import tensorflow as tf

model = tf.keras.models.load_model("hawaiian_fish_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("hawaiian_fish_model.tflite", "wb") as f:
    f.write(tflite_model)
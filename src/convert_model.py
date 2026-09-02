from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
model = tf.keras.models.load_model(PROJECT_ROOT / 'hawaiian_fish_model.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(PROJECT_ROOT / 'hawaiian_fish_model.tflite', 'wb') as f:
    f.write(tflite_model)
from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'hawaiian_fish_model.tflite'
IMAGE_PATH = PROJECT_ROOT / 'test_img_2.jpg'
LABELS_PATH = PROJECT_ROOT / 'labels.txt'

with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = tf.lite.Interpreter(model_path = MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open(IMAGE_PATH).convert('RGB').resize((224, 224))
image_array = np.array(image, dtype=np.float32)
image_array = image_array / 127.5 - 1.0
image_array = np.expand_dims(image_array, axis=0)

input_dtype = input_details[0]['dtype']
if np.issubdtype(input_dtype, np.integer):
    input_scale, input_zero_point = input_details[0]['quantization']
    img = np.round(image_array / input_scale + input_zero_point).astype(input_dtype)
else:
    img = image_array.astype(input_dtype)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

predictions = interpreter.get_tensor(output_details[0]['index'])[0]
if np.issubdtype(output_details[0]['dtype'], np.integer):
    output_scale, output_zero_point = output_details[0]['quantization']
    predictions = (predictions.astype(np.float32) - output_zero_point) * output_scale

top_index = np.argmax(predictions)

print(output_details)

print('Prediction:', labels[top_index])
print('Confidence:', predictions[top_index])

top_3 = np.argsort(predictions)[-3:][::-1]

for index in top_3:
    print(
        f"{labels[index]}: "
        f"{predictions[index] * 100:.2f}%"
    )
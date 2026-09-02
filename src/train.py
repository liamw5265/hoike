from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
TRAIN_DIR = PROJECT_ROOT / 'data' / 'train'
VAL_DIR = PROJECT_ROOT / 'data' / 'val'


def contains_images(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    return any(
        path.suffix.lower() in image_extensions
        for path in directory.rglob('*')
        if path.is_file()
    )


has_validation_images = contains_images(VAL_DIR)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2 if not has_validation_images else None,
    subset='training' if not has_validation_images else None,
    seed=SEED,
)

class_names = train_ds.class_names
num_classes = len(class_names)

if has_validation_images:
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=class_names,
        shuffle=False,
    )
else:
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=SEED,
        class_names=class_names,
        shuffle=False,
    )

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
)

base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.Rescaling(1 / 127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=4, restore_best_weights=True
        )
    ],
)

model.save(PROJECT_ROOT / 'hawaiian_fish_model.keras')

with open(PROJECT_ROOT / 'labels.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')
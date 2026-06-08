import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

val_ds  = tf.keras.utils.image_dataset_from_directory(
    "data/val",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

base_model = tf.keras.applications.MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False, # Remove original final layer to allow custom final layer
    weights = "imagenet"
)

# Preserve pre-trained neural net
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation = "softmax")
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10
)

model.save("hawaiian_fish_model.keras")

with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")
# Transfer Learning in TensorFlow

**Senior-Level Explanation + Practical Code**

------------------------------------------------------------------------

## 📌 What Is Transfer Learning?

Transfer learning is a deep learning technique where a model developed
for one task is reused as the starting point for a model on a second
task.

Instead of training a neural network from scratch, we:

1.  **Take a pre-trained model** (trained on a large dataset like
    ImageNet)
2.  **Freeze its layers** (to keep learned features)
3.  **Add new layers** for our custom task
4.  **Train only the new layers** on the smaller dataset

This approach: - Saves **training time** - Works well with **limited
data** - Provides **higher accuracy** by using learned features such as
edges, textures, shapes.

------------------------------------------------------------------------

## 📌 When to Use Transfer Learning

✔ You have **small or medium-sized dataset**\
✔ You want **high accuracy quickly**\
✔ You work with **image, NLP, or audio tasks**\
✔ You want to **avoid long training times**

------------------------------------------------------------------------

## 📌 Popular Pre‑trained Models in TensorFlow

TensorFlow provides many models under `tf.keras.applications`,
including:

-   **MobileNetV2** -- lightweight and fast\
-   **VGG16 / VGG19** -- older but powerful\
-   **ResNet50 / ResNet101** -- deeper models with skip connections\
-   **EfficientNet** -- state‑of‑the‑art performance

In this file, we use **MobileNetV2** because it is fast and recommended
for transfer learning on small datasets.

------------------------------------------------------------------------

# 🚀 Practical Example: Transfer Learning in TensorFlow (Image Classification)

Here is a complete, clean TensorFlow code for transfer learning.

------------------------------------------------------------------------

## ✅ Step 1: Import Libraries

``` python
import tensorflow as tf
from tensorflow.keras import layers, models
```

------------------------------------------------------------------------

## ✅ Step 2: Load Dataset

Example: Load images from a directory

``` python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/val",
    image_size=(224, 224),
    batch_size=32
)
```

------------------------------------------------------------------------

## ✅ Step 3: Load Pre-trained MobileNetV2

``` python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
```

-   `include_top=False` removes the original classifier
-   `weights="imagenet"` loads pre-trained ImageNet knowledge

------------------------------------------------------------------------

## ✅ Step 4: Freeze the Base Model

``` python
base_model.trainable = False
```

This prevents destroying pre‑trained features.

------------------------------------------------------------------------

## ✅ Step 5: Add Custom Layers

``` python
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(3, activation="softmax")(x)   # Example: 3 classes

model = tf.keras.Model(inputs, outputs)
```

------------------------------------------------------------------------

## ✅ Step 6: Compile the Model

``` python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

------------------------------------------------------------------------

## ✅ Step 7: Train

``` python
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)
```

------------------------------------------------------------------------

# 🎯 Optional: Fine‑Tuning (Advanced)

After initial training:

``` python
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # very small LR
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds
)
```

Fine-tuning improves accuracy by adjusting the deeper layers slightly.

------------------------------------------------------------------------

# 📊 Final Notes

-   Transfer learning **dramatically reduces training time**
-   Works extremely well even on **CPU**
-   Fine‑tuning should always be done with a **very small learning
    rate**

------------------------------------------------------------------------

# ✅ Summary

  Step   Description
  ------ ------------------------------------
  1      Load dataset
  2      Load pre-trained model
  3      Freeze base layers
  4      Add classification head
  5      Train
  6      (Optional) Fine‑tune deeper layers

------------------------------------------------------------------------

Feel free to extend this file with your own dataset experiments.

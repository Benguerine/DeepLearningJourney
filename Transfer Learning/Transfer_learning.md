# Transfer Learning

Transfer learning is a **machine learning technique** where a model
trained on one task is **reused** (fully or partially) on a **different
but related task**. Instead of training from scratch, we *transfer* the
learned knowledge.

------------------------------------------------------------------------

## **Why Transfer Learning?**

Traditional deep learning requires: - **Large datasets** - **High
computational cost** - **Long training time**

Transfer learning solves this by: - Using a **pre-trained model**
(trained on millions of images) - **Adapting** it to your new dataset

------------------------------------------------------------------------

## Key Concepts

### **1. Pre-trained Model**

A **model already trained** on a large benchmark dataset (e.g.,
ImageNet).\
Example: **VGG16**, **ResNet50**, **MobileNet**.

### **2. Feature Extraction**

We use the **pre-trained layers as a fixed feature extractor**\
--> Only train a new classifier on top.

### **3. Fine-Tuning**

We **unfreeze some deeper layers** of the pre-trained model\
--> Allow the model to adjust to the new task.

### **4. Freezing Layers**

"**Freezing**" means preventing a layer's weights from updating during
training.

------------------------------------------------------------------------

## Example Real-Life Analogy

Imagine you already know **English**.\
Learning **Spanish** becomes easier because: - Many words look similar\
- The alphabet is the same

This is exactly like transfer learning:\
You already have knowledge → You reuse it → You learn faster.

------------------------------------------------------------------------

# Transfer Learning in TensorFlow (Code + Explanation)

Below is a professional-level example using **MobileNetV2**.

------------------------------------------------------------------------

## Step-by-Step Code (TensorFlow)

``` python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load pre-trained MobileNetV2 without the top classifier
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model (no training)
base_model.trainable = False

# Build the classifier head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # For 10 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

------------------------------------------------------------------------

# Fine‑Tuning the Model

After training the classifier head,\
we can **unfreeze** deeper layers for improved accuracy.

``` python
# Unfreeze last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

------------------------------------------------------------------------

# Best Practices 

### **Use lower learning rates**

Fine-tuning should use **small LR** to avoid destroying learned
features.

### **Avoid unfreezing too many layers**

May cause **overfitting** if your dataset is small.

### **Use data augmentation**

Improves generalization.

### **Use early stopping**

Prevents unnecessary training.

------------------------------------------------------------------------

# Summary

  Concept                 Meaning
  ----------------------- ------------------------------------
  **Transfer Learning**   Reusing a model's learned features
  **Pre-trained Model**   Model trained on huge dataset
  **Freezing**            Stop weights from training
  **Fine-Tuning**         Unfreezing deeper layers

------------------------------------------------------------------------

# Final Thought

Transfer learning allows you to build **high‑accuracy models** with
**low data**, **low computation**, and **faster training**.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

print("Starting Diabetic Retinopathy Model Training...")

# 1. Generate synthetic dataset (since we don't have real data for quick demo)
def create_synthetic_data(num_samples=10000):
    """Create synthetic retinal images for demonstration"""
    X = []
    y = []

    for i in range(num_samples):
        # Create a fake retinal image (224x224x3)
        img = np.random.rand(224, 224, 3).astype(np.float32)

        # Add some patterns to make it look medical-ish
        # Circular mask (retina shape)
        jj, kk = np.meshgrid(np.arange(224), np.arange(224), indexing='ij')
        mask = ((jj - 112) ** 2 + (kk - 112) ** 2) < 100 ** 2
        img[mask] = img[mask] * 0.7 + 0.3  # Brighter inside

        X.append(img)

        # Random label (0-4) - diabetic retinopathy severity
        y.append(np.random.randint(0, 5))

    return np.array(X), np.array(y)


print("Creating synthetic dataset...")
X, y = create_synthetic_data(10000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# 2. Build the model
print("Building model architecture...")


def create_model():
    # Use MobileNetV2 (lighter than ResNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model
    base_model.trainable = False

    # Add custom layers
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(5, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    return model


model = create_model()

# 3. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Train model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32,
    verbose=1
)

# 5. Save model
model.save('diabetic_retinopathy_model.h5')
print("Model saved as 'diabetic_retinopathy_model.h5'")

# 6. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

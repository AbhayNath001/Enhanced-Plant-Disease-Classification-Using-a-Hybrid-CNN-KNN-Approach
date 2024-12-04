import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

# Load the datasets
# Placeholder for training and testing images and labels (to be replaced with actual dataset loading)
train_images = np.random.rand(6400, 224, 224, 3)  # Example image data
train_labels = np.random.randint(0, 8, size=6400)  # Example labels for Mango Leaf BD dataset
test_images = np.random.rand(1600, 224, 224, 3)  # Example test data
test_labels = np.random.randint(0, 8, size=1600)  # Example labels

# Data Augmentation setup
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.15, 0.35]
)

# Apply data augmentation to training images
train_gen = datagen.flow(train_images, train_labels, batch_size=16)

# CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'),
    layers.AvgPool2D(pool_size=(3, 3), strides=2),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.AvgPool2D(pool_size=(3, 3), strides=2),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.AvgPool2D(pool_size=(3, 3), strides=2),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(8, activation='softmax')  # 8 classes for Mango Leaf BD dataset
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the CNN model
model.fit(train_gen, epochs=35, steps_per_epoch=len(train_images) // 16)

# To extract features for KNN, remove the final softmax layer
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract CNN features for KNN classifier
cnn_features_train = feature_extractor.predict(train_images)  # Features from the training set
cnn_features_test = feature_extractor.predict(test_images)   # Features from the test set

# Use KNN to classify based on CNN features
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(cnn_features_train, train_labels)
knn_predictions = knn.predict(cnn_features_test)

# Evaluation Metrics
accuracy = accuracy_score(test_labels, knn_predictions)
precision = precision_score(test_labels, knn_predictions, average='weighted')
recall = recall_score(test_labels, knn_predictions, average='weighted')
f1 = f1_score(test_labels, knn_predictions, average='weighted')

# Output the evaluation metrics
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")

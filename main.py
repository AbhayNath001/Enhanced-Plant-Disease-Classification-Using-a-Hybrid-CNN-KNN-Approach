import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
import kagglehub

dataset_dir = kagglehub.dataset_download("aryashah2k/mango-leaf-disease-dataset")

classes = ['Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil', 
           'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Red_Rust', 'Sooty_Mould']

img_size = 224

def load_images_and_labels(dataset_dir, classes, img_size):
    images, labels = [], []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(img_size, img_size))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
    images, labels = np.array(images), np.array(labels)
    return images, labels

print("Loading dataset...")
images, labels = load_images_and_labels(dataset_dir, classes, img_size)

images, labels = shuffle(images, labels, random_state=42)

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

train_images = train_images / 255.0
test_images = test_images / 255.0

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.15, 0.35]
)

train_gen = datagen.flow(train_images, train_labels, batch_size=16)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3), padding='same'),
    layers.AvgPool2D(pool_size=(3, 3), strides=2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.AvgPool2D(pool_size=(3, 3), strides=2),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.AvgPool2D(pool_size=(3, 3), strides=2),

    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(8, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training the CNN model...")
model.fit(train_gen, epochs=35, steps_per_epoch=len(train_images) // 16, verbose=1)

feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

print("Extracting features...")
cnn_features_train = feature_extractor.predict(train_images)
cnn_features_test = feature_extractor.predict(test_images)

print("Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(cnn_features_train, train_labels)

knn_predictions = knn.predict(cnn_features_test)

accuracy = accuracy_score(test_labels, knn_predictions)
precision = precision_score(test_labels, knn_predictions, average='weighted')
recall = recall_score(test_labels, knn_predictions, average='weighted')
f1 = f1_score(test_labels, knn_predictions, average='weighted')

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")

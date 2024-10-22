import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Parameters
image_dir = 'captcha/train'
input_shape = (50, 200, 1)
num_classes = 62 # number of characters
epochs = 100
batch_size = 32
activation = 'relu'

labels=[]

# Load and preprocess data
def load_data(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = load_img(image_path, color_mode='grayscale', target_size=(50, 200))
            image = img_to_array(image)
            image = image / 255.0
            images.append(image)
            labels.append(filename.split('.')[0])
    return np.array(images), labels

images, labels = load_data(image_dir)

# Encode labels
def encode_labels(labels, num_classes):
    label_map = {char: idx for idx, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')}
    encoded_labels = [to_categorical([label_map[char] for char in label], num_classes=num_classes) for label in labels]
    return np.array(encoded_labels)

encoded_labels = encode_labels(labels, num_classes)

# Define the model
def build_model(input_shape, num_classes, num_characters):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation=activation, padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation=activation, padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation=activation, padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.RepeatVector(num_characters)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)

    model = models.Model(inputs, x)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

num_characters = 5
model = build_model(input_shape, num_classes, num_characters)

# Train the model
model.fit(images, encoded_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Save the model
model_save_path = 'captcha_solver_model.h5'
model.save(model_save_path)
print(f'Model saved to {model_save_path}')

# Evaluate the model
loss, accuracy = model.evaluate(images, encoded_labels)
print(f'Loss: {loss}, Accuracy: {accuracy}')


import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
from tensorflow.keras import layers, models, callbacks, utils

batch_size = 16
input_shape = (50, 200, 1)
num_classes = 62

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=5,  # Random rotation degrees
    width_shift_range=0.1,  # Horizontal shift
    height_shift_range=0.1,  # Vertical shift
    shear_range=0.2,  # Shear intensity
    zoom_range=0.2  # Zoom in/out
)

train_image_dir = 'captcha/train'
val_image_dir = 'captcha/test'

train_files = [f for f in os.listdir(train_image_dir) if f.endswith('.png')]
val_files = [f for f in os.listdir(val_image_dir) if f.endswith('.png')]

def custom_generator(file_list, image_dir, datagen, batch_size, input_shape, num_classes):
    label_map = {char: idx for idx, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')}
    images = []
    labels = []
    image_count = 0

    while True:
        np.random.shuffle(file_list)
        for filename in file_list:
            image_path = os.path.join(image_dir, filename)
            image = load_img(image_path, color_mode='grayscale', target_size=input_shape[:2])
            image = img_to_array(image)
            augmented_image = datagen.random_transform(image)  # Apply augmentation
            augmented_image /= 255.0

            # Save augmented image
            aug_image_filename = f"aug_{image_count}_{filename}"
            aug_image_path = os.path.join(image_dir, aug_image_filename)
            save_img(aug_image_path, augmented_image)
            image_count += 1

            label = filename.split('.')[0]  # Assuming the filename is the label

            # Convert label to one-hot encoding
            encoded_label = [utils.to_categorical(label_map[char], num_classes=num_classes) for char in label]

            images.append(augmented_image)
            labels.append(encoded_label)

            if len(images) == batch_size:
                yield np.array(images), np.array(labels)
                images, labels = [], []

# Generators for training and validation
train_generator = custom_generator(train_files, train_image_dir, datagen, batch_size, input_shape, num_classes)
validation_generator = custom_generator(val_files, val_image_dir, datagen, batch_size, input_shape, num_classes)

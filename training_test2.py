import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import layers, models, callbacks, utils
import os

# Parameters
input_shape = (50, 200, 1)
num_classes = 62  # A-Z, a-z, 0-9 (62 classes)
batch_size = 16
max_length = 5  # Assuming the CAPTCHA length is 5 characters
activation = 'relu'  # Use a valid activation function
epochs = 100

# Create a mapping for characters to integers
label_map = {char: idx for idx, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')}


# Function to extract and preprocess labels
def extract_label(filename):
    if filename.startswith('aug_numbers_'):
        label = filename.split('aug_numbers_')[1].split('.')[0]
    else:
        label = filename.split('.')[0]
    return label.replace('_', '')


def encode_labels(labels, num_classes):
    encoded_labels = []
    for label in labels:
        # Convert each element of label to a string if it's not already one
        if isinstance(label, list) or isinstance(label, np.ndarray):
            label_str = ''.join([str(char) for char in label])
        else:
            label_str = str(label)

        # Remove underscores from the label string
        cleaned_label = label_str.replace('_', '')

        # Encode each character in the cleaned label
        encoded_label = [utils.to_categorical(label_map.get(char, 0), num_classes=num_classes) for char in cleaned_label]
        encoded_labels.append(encoded_label)

    return encoded_labels


# Custom generator to yield images and labels with padding
def custom_generator(file_list, label_list, image_dir, batch_size, input_shape, num_classes, max_length):
    while True:
        np.random.shuffle(file_list)
        images, labels = [], []
        for filename, label in zip(file_list, label_list):
            image_path = os.path.join(image_dir, filename)
            image = load_img(image_path, color_mode='grayscale', target_size=input_shape[:2])
            image = img_to_array(image) / 255.0

            # Convert label to one-hot encoded sequences
            encoded_label = encode_labels([label], num_classes)[0]

            # Pad the encoded label to `max_length`
            if len(encoded_label) < max_length:
                padding = [np.zeros(num_classes) for _ in range(max_length - len(encoded_label))]
                encoded_label.extend(padding)
            else:
                encoded_label = encoded_label[:max_length]

            images.append(image)
            labels.append(encoded_label)

            if len(images) == batch_size:
                yield np.array(images), np.array(labels)
                images, labels = [], []


# File paths
train_image_dir = 'captcha_new'
val_image_dir = 'captcha_new/test'

# Load filenames and extract labels
train_files = [f for f in os.listdir(train_image_dir) if f.endswith('.png')]
train_labels = [extract_label(f) for f in train_files]

val_files = [f for f in os.listdir(val_image_dir) if f.endswith('.png')]
val_labels = [extract_label(f) for f in val_files]

# Generators for training and validation
train_generator = custom_generator(train_files, train_labels, train_image_dir, batch_size, input_shape, num_classes, max_length)
validation_generator = custom_generator(val_files, val_labels, val_image_dir, batch_size, input_shape, num_classes, max_length)


# Define the model
def build_model(input_shape, num_classes, num_characters, dropout_rate):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation=activation, padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation=activation, padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation=activation, padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation=activation, padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.RepeatVector(num_characters)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)

    model = models.Model(inputs, x)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


num_characters = 5  # Adjust according to your CAPTCHA length
model = build_model(input_shape, num_classes, num_characters, 0.5)


# Custom callback to save the model when accuracy reaches 1.0000 for 5 epochs
class SaveOnAccuracy(callbacks.Callback):
    def __init__(self, save_path, patience=5):
        super(SaveOnAccuracy, self).__init__()
        self.save_path = save_path
        self.patience = patience
        self.best_acc = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get('accuracy')
        if current_acc == 1.0:
            self.wait += 1
            if self.wait >= self.patience:
                print(f'\nAccuracy is 1.0000 for {self.patience} epochs. Saving model to {self.save_path}.')
                self.model.save(self.save_path)
        else:
            self.wait = 0


# Define the save path for the model
model_save_path = 'captcha_solver_model.h5'

# Create an instance of the custom callback
save_callback = SaveOnAccuracy(save_path=model_save_path, patience=5)

# Train the model with the generator and callback
steps_per_epoch = len(train_files) // batch_size
validation_steps = len(val_files) // batch_size

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[save_callback]
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')
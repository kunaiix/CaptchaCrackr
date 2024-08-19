import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load the model for inference

model_save_path = 'captcha_solver_model_1.h5'
model = load_model(model_save_path)
print('Model loaded successfully')


def predict_captcha(model, image_path):
    image = load_img(image_path, color_mode='grayscale', target_size=(50, 200))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_labels = np.argmax(predictions, axis=-1)

    idx_to_char = {idx: char for idx, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')}
    predicted_text = ''.join([idx_to_char[idx] for idx in predicted_labels[0]])

    return predicted_text

# Example usage
new_captcha_image_path = 'captcha/test/2nf26.png'
predicted_text = predict_captcha(model, new_captcha_image_path)
print(f'Predicted text: {predicted_text}')

# optimizers
# activation
# try different preprocessing techniques
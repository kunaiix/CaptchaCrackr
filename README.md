# CaptchaCrackr
A python program which can solve captcha images.

Made using Tensorflow, Keras, Keras-OCR and Numpy. The model is trained using a captcha image dataset from [https://github.com/AakashKumarNain/CaptchaCracker]. 

Developed with 5 character captcha images, uses 'relu' activator and 'adam' optimizer.
Image preprocessing is basic, converts to greyscale and enhances it. 
Image target size is 200x50

The goal of this project is to build an api where the model is hosted. Any 5 character captcha should be solved with high accuracy. Will focus on training and improving the model to reach this goal.

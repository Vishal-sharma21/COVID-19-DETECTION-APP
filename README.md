COVID-19 Detection App

This project is a COVID-19 Detection Web Application developed using Python, Streamlit, and Deep Learning techniques. The application analyzes chest X-ray images and predicts whether a patient is COVID-19 positive or negative.


Multiple deep learning approaches were implemented and compared:

Basic CNN Model
Transfer Learning using VGG16
Data Augmentation + VGG16

The best-performing model was selected and integrated into the Streamlit application for real-time predictions.

Features:-
Upload Chest X-ray images for prediction
User-friendly Streamlit web interface
Fast and accurate COVID-19 detection

Application Workflow:-
Upload a chest X-ray image.
The image is preprocessed.
The trained deep learning model analyzes the image.
The application displays the prediction:
COVID-19:-
Positive
Normal

Technologies Used:-
Python
TensorFlow 
Streamlit
NumPy
Pandas
Matplotlib
OpenCV
Scikit-learn

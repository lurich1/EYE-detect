from flask import Flask, render_template, request
import os
import keras
from PIL import Image
import numpy as np
import cv2

# Initialize Flask app
app = Flask(__name__)

# Define function to return the class of the ear condition
def return_class(img_path):
    try:
        # Load the pre-trained model for ear conditions
        model = keras.models.load_model('main_model.h5')
        print(f"Image path: {img_path}")

        # Open and preprocess the image
        test_img = Image.open(img_path)
        test_img_array = np.asarray(test_img)
        test_img_resize = cv2.resize(test_img_array, (256, 256))
        test_img_reshape = np.reshape(test_img_resize, (1, 256, 256, 3))

        # Predict the class
        predictions = model.predict(test_img_reshape)
        classes = ["Chronic otitis media", "Ear wax plug", "Myringlorisis", "Normal ear drum"]
        predicted_class = classes[np.argmax(predictions[0])]

        return predicted_class
    except Exception as e:
        print(f"Error in return_class: {e}")
        return "Error in prediction"

def return_eye_class(img_path):
    try:
        # Load the pre-trained model for eye diseases
        eye_model = keras.models.load_model('model_eye.h5')
        print(f"Eye image path: {img_path}")

        # Open and preprocess the image
        eye_img = Image.open(img_path)
        eye_img_array = np.asarray(eye_img)
        print(f"Original image shape: {eye_img_array.shape}")
        eye_img_resize = cv2.resize(eye_img_array, (224, 224))
        print(f"Resized image shape: {eye_img_resize.shape}")
        eye_img_reshape = np.reshape(eye_img_resize, (1, 224, 224, 3))
        print(f"Reshaped image shape: {eye_img_reshape.shape}")

        # Predict the class
        eye_predictions = eye_model.predict(eye_img_reshape)
        print(f"Predictions: {eye_predictions}")
        eye_classes = ["Cataract", "diabetic_retinopathy", "Glaucoma", "Normal eye"]
        eye_predicted_class = eye_classes[np.argmax(eye_predictions[0])]

        return eye_predicted_class
    except Exception as e:
        print(f"Error in return_eye_class: {e}")
        return f"Error in prediction: {e}"


# Define the main route
@app.route('/')
def main():
    return render_template('main.html')

# Define the home route
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/eyeabout')
def eyeabout():
    return render_template('eyeabout.html')

# Define the about route
@app.route('/about')
def about():
    return render_template('about.html')

# Define the detection page route
@app.route('/dect')
def dect():
    return render_template('dect.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/indexeye')
def indexeye():
    return render_template('indexeye.html')

# Define the result route for ear conditions
@app.route('/result', methods=['POST'])
def result():
    try:
        if request.method == 'POST':
            input_image = request.files['input_image']
            print(f"Uploaded file: {input_image.filename}")
            save_path = os.path.join('static', input_image.filename)
            input_image.save(save_path)
            output = return_class(save_path)
            return render_template("index.html", img_path=input_image.filename, output=output)
    except Exception as e:
        print(f"Error in result function: {e}")
        return "An error occurred during the prediction process."

# Define the result route for eye diseases
@app.route('/result_eye', methods=['POST'])
def result_eye():
    try:
        if request.method == 'POST':
            input_image = request.files['input_image']
            print(f"Uploaded file: {input_image.filename}")
            save_path = os.path.join('static', input_image.filename)
            input_image.save(save_path)
            output = return_eye_class(save_path)
            return render_template("indexeye.html", img_path=input_image.filename, output=output)
    except Exception as e:
        print(f"Error in result_eye function: {e}")
        return f"An error occurred during the prediction process: {e}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

model_paths = {
    'First run CNN': 'clothing_classification_model.keras',
    'Second run adjustments made': 'clothing_classification_fewer_classes_model_vgg16.keras',
    'Third run': 'clothing_classification_model_deep_cnn.keras',
    'Fourth run': 'training_custom_dataset_vgg16.keras'  
}

models = {}
for key, path in model_paths.items():
    print(f"Checking model path: {path}")  # Debugging information
    if os.path.exists(path):
        models[key] = load_model(path)
        print(f"Loaded model {key} from {path}")
    else:
        print(f"Model file {path} not found, skipping...")

# Print the current working directory and list of files for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Available files: {os.listdir('.')}")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class_names = ['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items', 'Sporting Goods', 'Home']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    if 'model' not in request.form:
        return jsonify({'error': 'No model selected'})

    file = request.files['file']
    model_key = request.form['model']

    print(f"Model selected by user: {model_key}")  # Debugging information
    print(f"Available models: {models.keys()}")  # Debugging information

    if model_key not in models:
        print(f"Invalid model selected: {model_key}")  # Debugging information
        return jsonify({'error': 'Invalid model selected'})

    model = models[model_key]

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Prepare the image for prediction
        image = load_img(filepath, target_size=(128, 128))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        # Predict
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]

        return jsonify({'predicted_class': predicted_class_name, 'probabilities': predictions.tolist()})

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)

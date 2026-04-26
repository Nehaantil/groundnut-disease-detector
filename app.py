from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import gdown

app = Flask(__name__)

# Google Drive file IDs
MODEL_FILE_ID = '16tn3KCyrWQiNLTE8ej7a4pau70jZozmX'
LABELS_FILE_ID = '1SMrVQjWRxO0tl3YKHasbIRBLjDizNm8c'

MODEL_PATH = 'model/disease_model.keras'
LABELS_PATH = 'model/class_labels.json'

# Download model files if not present
os.makedirs('model', exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_FILE_ID}', MODEL_PATH, quiet=False)

if not os.path.exists(LABELS_PATH):
    print("Downloading class labels from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={LABELS_FILE_ID}', LABELS_PATH, quiet=False)

# Load model and labels
model = tf.keras.models.load_model(MODEL_PATH,compile=False)

with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)

disease_info = {
    'early_leaf_spot_1': {
        'description': 'Early Leaf Spot is caused by the fungus Cercospora arachidicola.',
        'treatment': 'Apply Mancozeb or Chlorothalonil fungicide. Remove infected leaves.'
    },
    'early_rust_1': {
        'description': 'Early Rust is caused by Puccinia arachidis fungus.',
        'treatment': 'Spray Propiconazole fungicide. Ensure proper spacing between plants.'
    },
    'healthy_leaf_1': {
        'description': 'The leaf appears healthy with no visible disease.',
        'treatment': 'No treatment needed. Continue regular crop monitoring.'
    },
    'late_leaf_spot_1': {
        'description': 'Late Leaf Spot is caused by Phaeoisariopsis personata fungus.',
        'treatment': 'Apply Carbendazim or Tebuconazole. Rotate crops every season.'
    },
    'nutrition_deficiency_1': {
        'description': 'The plant shows signs of nutrient deficiency.',
        'treatment': 'Apply balanced NPK fertilizer. Test soil and adjust nutrients.'
    },
    'rust_1': {
        'description': 'Rust disease caused by Puccinia arachidis.',
        'treatment': 'Use resistant varieties. Apply fungicide at early stages.'
    }
}

def prepare_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    try:
        image = Image.open(file).convert('RGB')
        prepared = prepare_image(image)
        predictions = model.predict(prepared)
        predicted_index = str(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100
        class_name = class_labels[predicted_index]
        info = disease_info.get(class_name, {
            'description': 'No description available.',
            'treatment': 'Please consult an agricultural expert.'
        })
        if confidence < 70:
            warning = "Low confidence — please try a clearer image."
        else:
            warning = ""
        return jsonify({
            'disease': class_name.replace('_1', '').replace('_', ' ').title(),
            'confidence': round(confidence, 2),
            'description': info['description'],
            'treatment': info['treatment'],
            'warning': warning
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
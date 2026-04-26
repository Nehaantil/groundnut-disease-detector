from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import json
import os
import gdown

app = Flask(__name__)

MODEL_PATH = 'model/disease_model.tflite'
LABELS_PATH = 'model/class_labels.json'
KERAS_PATH = 'model/disease_model.keras'

os.makedirs('model', exist_ok=True)

# Download files from Google Drive
if not os.path.exists(KERAS_PATH):
    print("Downloading model...")
    gdown.download(
    f'https://drive.google.com/uc?id=16tn3KCyrWQiNLTE8ej7a4pau70jZozmX',
    KERAS_PATH, quiet=False
)

if not os.path.exists(LABELS_PATH):
    print("Downloading labels...")
    gdown.download(
    f'https://drive.google.com/uc?id=1SMrVQjWRxO0tl3YKHasbIRBLjDizNm8c',
    LABELS_PATH, quiet=False
)

# Convert to TFLite if not already done
if not os.path.exists(MODEL_PATH):
    print("Converting to TFLite...")
    import tensorflow as tf
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print("Conversion done!")

# Load TFLite model
try:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
except:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded!")

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
    image = np.array(image, dtype=np.float32) / 255.0
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
        interpreter.set_tensor(input_details[0]['index'], prepared)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
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
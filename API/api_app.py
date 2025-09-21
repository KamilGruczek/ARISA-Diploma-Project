import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, Blueprint, request, jsonify
from keras_facenet import FaceNet
from ARISA_DSML.resolve import detect_faces
from ARISA_DSML.helpers import FaceRecognizer
from PIL import Image
import numpy as np

embedder = FaceNet()
rec = FaceRecognizer(embedder)
rec.load()
print("Data loaded")

app = Flask(__name__)

face_detection = Blueprint('face_detection', __name__)

@face_detection.route('/detect_faces', methods=['POST'])
def detect_faces_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Process the image and detect faces
    image = Image.open(image_file.stream).convert('RGB')
    image_np = np.array(image)
    face_locations = detect_faces(image_np)
    
    return jsonify({'face_locations': face_locations}), 200

train_face_recognition = Blueprint('train_face_recognition', __name__)

@train_face_recognition.route('/train', methods=['POST'])
def train():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Image and name are required'}), 400

    image_file = request.files['image']
    name = request.form['name']

    image = Image.open(image_file.stream).convert('RGB')
    image = image.resize((160, 160))
    img = np.array(image)
    success = rec.add_person([img], [name])
    rec.save()

    if success:
        return jsonify({'message': 'Model trained successfully'}), 200
    else:
        return jsonify({'error': 'Failed to train model'}), 500

predict_face = Blueprint('predict_face', __name__)

@predict_face.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert('RGB')
    image = image.resize((160, 160))
    img = np.array(image)
    
    label, confidence = rec.predict([img], [0])
    if len(label) == 0:
        return jsonify({"error": "No face detected"}), 400

    return jsonify({'predicted_name': str(label[0]), 'confidence': float(confidence[0])}), 200

# Register endpoints
app.register_blueprint(face_detection)
app.register_blueprint(train_face_recognition)
app.register_blueprint(predict_face)

if __name__ == '__main__':
    app.run(debug=True)
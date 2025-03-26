from flask import Flask, request, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = './output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Posizione base e step incrementale per far "muovere" il robot ad ogni richiesta
default_joints = np.array([0.0, -1.16, -0.0, -2.3, -0.0, 1.6, 1.1, 0.4, 0.4])
step = np.array([0.05, -0.03, 0.02, 0.04, -0.02, 0.01, -0.03, 0.01, 0.01])
current_step = 0

@app.route('/process_image', methods=['POST'])
def process_image():
    global current_step

    if 'image' not in request.files:
        return jsonify({'error': 'Nessuna immagine trovata'}), 400

    image = request.files['image']
    description = request.form.get('description', 'no description')

    # Salvataggio immagine
    filename = f'image_{current_step:04d}.png'
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    image.save(filepath)
    print(f"✅ Salvata immagine: {filename} | Descrizione: {description}")

    # Calcolo posizione joint da inviare
    joint_position = (default_joints + step * current_step).tolist()
    current_step += 1

    return jsonify({'joint_position': joint_position})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

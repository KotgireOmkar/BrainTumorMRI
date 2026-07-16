"""
Lightweight preview server - serves the frontend UI without TensorFlow.
Prediction endpoints return mock data so the full UI can be explored.
"""
import os
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'Frontend & Core', 'templates'),
    static_folder=os.path.join(BASE_DIR, 'Frontend & Core', 'static')
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({
        'success': True,
        'class': 'Glioblastoma T1C+',
        'confidence': 98.4,
        'confidence_level': 'Very High',
        'confidence_color': 'success',
        'image': '',
        'dicom_metadata': {
            'sequence_type': 'T1-Weighted Contrast Enhanced (T1C+)',
            'tr': '450 ms', 'te': '15 ms', 'flip_angle': '90°',
            'contrast_agent': 'Gadolinium (Gd-DTPA)',
            'magnetic_field': '3.0 Tesla',
            'scanner_model': 'Siemens MAGNETOM Skyra',
            'coil_type': '16-Channel Head Coil',
            'slice_thickness': '5.0 mm',
            'pixel_spacing': [0.45, 0.45],
            'patient_name': 'Preview Patient',
            'patient_id': 'NAI-PREVIEW',
            'patient_age': '45',
            'study_date': '2026-07-15'
        },
        'top_predictions': [
            {'class': 'Glioblastoma T1C+', 'confidence': 98.4},
            {'class': 'Astrocytoma T2', 'confidence': 0.8},
            {'class': 'Oligodendroglioma T2', 'confidence': 0.4},
            {'class': 'Meningioma T1', 'confidence': 0.2},
            {'class': 'Ependymoma T1C+', 'confidence': 0.1}
        ]
    })

@app.route('/chat', methods=['POST'])
def chat():
    return jsonify({'response': '### Preview Mode 🔬\nThe AI assistant is running in **preview mode** (no TensorFlow). The full AI backend requires TensorFlow DLLs to be unblocked by your system policy.\n\nThe UI you see is fully functional — upload, chat, and navigation all work.'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy (preview)', 'classes': 44}), 200

if __name__ == '__main__':
    print("=" * 50)
    print("  NeuroAI Preview Server (no TensorFlow)")
    print("  http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=False, host='127.0.0.1', port=5000)

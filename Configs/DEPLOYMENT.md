# Brain Tumor MRI Classifier - Web Dashboard 🧠

A modern **AI-powered web application** for detecting brain tumors in MRI images using deep learning. Built with Flask, TensorFlow, and EfficientNetV2.

## 🚀 Live Demo

**[View on Render](https://brain-tumor-classifier.onrender.com)** (Deploy your own!)

## ✨ Features

- 🔍 **44 Brain Tumor Classifications** - Detects Glioma, Meningioma, Pituitary, and more
- 🎨 **Modern Web UI** - Beautiful gradient design with smooth animations
- 📤 **Drag & Drop Upload** - Easy image selection with drag-and-drop support
- ⚡ **Real-time Analysis** - Instant predictions with confidence scores
- 📊 **Confidence Indicators** - Color-coded confidence levels (Success/Warning/Info)
- 🌐 **Fully Responsive** - Works perfectly on desktop, tablet, and mobile
- 🚀 **Cloud Hosted** - Deploy on Render with one click

## 🛠️ Tech Stack

- **Backend**: Flask, Python 3.11
- **ML Model**: TensorFlow, EfficientNetV2B0
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Hosting**: Render.com
- **Image Processing**: OpenCV, Pillow

## 📦 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Brain-Tumor-MRI-Classifier.git
   cd Brain-Tumor-MRI-Classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   > Navigate to `http://localhost:5000`

## 🌐 Deploy on Render

### Step 1: Prepare for Deployment

The repository includes all necessary files:
- `Procfile` - Specifies how to run the app
- `runtime.txt` - Python version
- `requirements.txt` - Dependencies

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/Brain-Tumor-MRI-Classifier.git
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to [Render.com](https://render.com)
2. Sign up or log in with GitHub
3. Click **"New +"** → **"Web Service"**
4. Select your repository
5. **Configuration**:
   - **Name**: `brain-tumor-classifier`
   - **Environment**: `Python 3.11`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free or Paid

6. Click **"Deploy"**
7. Wait for build (usually 3-5 minutes)
8. Your app is live! 🎉

### Access Your Deployment

Your app will be available at: `https://brain-tumor-classifier.onrender.com`

## 📱 API Endpoints

### POST `/predict`
Upload an MRI image and get predictions

**Request**:
```
Content-Type: multipart/form-data
file: <image_file>
```

**Response**:
```json
{
    "success": true,
    "class": "Glioblastoma T1",
    "confidence": 92.45,
    "confidence_level": "Very High",
    "confidence_color": "success",
    "image": "data:image/png;base64,..."
}
```

### GET `/health`
Health check endpoint

**Response**:
```json
{
    "status": "healthy",
    "classes": 44
}
```

## 🖼️ Usage

1. **Select or drag-drop an MRI image** (JPG, JPEG, PNG)
2. **Wait for analysis** (animated loading state)
3. **View results** with:
   - Tumor classification
   - Confidence percentage
   - Color-coded confidence level
   - Original image preview

## 📊 Model Architecture

- **Base Model**: EfficientNetV2B0 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen base with custom classification head
- **Input Size**: 224×224 pixels
- **Output Classes**: 44 brain tumor types across T1, T1C+, T2 MRI sequences

## 🎨 Design Features

- **Gradient Background**: Smooth blue-to-dark gradient
- **Modern UI Components**: Rounded buttons, hover effects, animations
- **Loading States**: Smooth spinner with status updates
- **Responsive Layout**: Mobile-first design
- **Color Coding**: Success (green), Warning (orange), Info (blue)

## 🔒 Security

- File size limit: 16MB
- Supported formats: JPEG, PNG
- Secure filename handling
- Server-side validation
- No data storage (temporary files deleted after analysis)

## ⚠️ Disclaimer

This tool is for **educational and demonstration purposes only**. It should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.

## 📝 Dataset

Trained on: [Brain Tumor MRI Images - Kaggle](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)

Classes:
- Astrocytoma, Carcinoma, Ependymoma, Ganglioglioma
- Germinoma, Glioblastoma, Granuloma, Medulloblastoma
- Meningioma, Neurocytoma, Oligodendroglioma, Papilloma
- Schwannoma, Tuberculoma, No Tumor
- Each in T1, T1C+, T2 MRI sequences

## 🚀 Performance

- **Model Accuracy**: ~95% on validation set
- **Inference Time**: ~2-3 seconds per image
- **File Upload Speed**: Instant (drag & drop)
- **Response Time**: < 5 seconds (including network latency)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests

## 📄 License

MIT License - see LICENSE file

## 📧 Contact

For questions or feedback, reach out at: [your-email@example.com](mailto:your-email@example.com)

## 🙏 Acknowledgments

- TensorFlow & Keras teams
- EfficientNet researchers
- Kaggle community for the dataset
- Render.com for hosting

---

**Made with ❤️ for medical AI education**

![Brain Tumor Classifier](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

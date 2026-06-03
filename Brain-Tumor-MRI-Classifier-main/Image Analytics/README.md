# 🩻 Image Analytics Module

This directory serves as the documentation and organization hub for the **Image Analytics** capabilities of the NeuroAI Radiology Workstation.

## 🛠️ Technology Stack & Libraries

The image analytics operations are powered by:
1. **pydicom (2.4.3)**: Standard library for opening, reading, and parsing medical DICOM (`.dcm`) image format data.
2. **OpenCV (4.8.1.78)**: Used for high-speed image processing, including grayscale conversions, canvas image adjustments, and resizing.
3. **Pillow (10.0.0)**: Used for high-fidelity image serialization, format conversions, and Base64 buffer outputs.

## 📂 Implementation Details

To maintain a standard web application architecture and client-server paradigm, the analytics code is integrated into:
* **Backend Processing (`Backend/app.py`)**: 
  - Parses incoming `.dcm` files to extract patient metadata, acquisition tags (TR, TE, flip angle), and pixel spacing properties.
  - Normalizes raw 12-bit/16-bit DICOM pixel arrays to 8-bit grayscale using Min-Max scaling.
  - Resizes images to $224 \times 224 \times 3$ to feed into the deep learning classifier.
* **Frontend Visualization (`Frontend & Core/static/js/script.js`)**:
  - Direct canvas manipulation to perform real-time client-side filter operations:
    - **Sobel Edge Detection** (boundary outlines).
    - **Thermal Pseudo-Color Mapping** (density mapping).
    - **Invert Colors** (high contrast).
    - **Brightness & Contrast Adjustments**.
  - **Caliper Ruler**: Interactively calculates physical distance in millimeters ($mm$) utilizing DICOM pixel spacing data and estimates tumor spheroid volume in cubic centimeters ($cm^3$).
  - **Line Profiler**: Samples voxel intensity across custom margins and graphs tissue density profiles in real-time.

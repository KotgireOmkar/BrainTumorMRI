#!/bin/bash
# Build script for Render
echo "=== Brain Tumor MRI Classifier Build Script ==="

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build complete!"
echo "🚀 Ready to deploy on Render"

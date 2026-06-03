import os
import numpy as np
import cv2
import tensorflow as tf
import warnings
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas, Frame
from PIL import Image, ImageTk
import threading
import time

warnings.filterwarnings("ignore")

# ------------------ CONSTANTS ------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Machine Learning', 'model', 'best_mri_classifier.h5'))

# Modern Color Palette
PRIMARY_COLOR = "#0F3460"      # Deep blue
SECONDARY_COLOR = "#16213E"    # Darker blue
ACCENT_COLOR = "#E94560"       # Pink/Red accent
LIGHT_COLOR = "#F0F0F0"        # Light gray
TEXT_COLOR = "#FFFFFF"         # White
SUCCESS_COLOR = "#2ECC71"      # Green
WARNING_COLOR = "#F39C12"      # Orange

# Load model by reconstructing the architecture and loading weights
# This avoids Keras deserialization version conflict bugs
def load_mri_model(weights_path):
    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(44, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return model

model = load_mri_model(MODEL_PATH)

# 44 class names from Kaggle dataset: fernando2rad/brain-tumor-mri-images-44c
# Sorted alphabetically (matches TF image_dataset_from_directory folder order)
CLASS_NAMES = [
    'Astrocytoma_T1',    'Astrocytoma_T1C+',  'Astrocytoma_T2',
    'Carcinoma_T1',      'Carcinoma_T1C+',    'Carcinoma_T2',
    'Ependymoma_T1',     'Ependymoma_T1C+',   'Ependymoma_T2',
    'Ganglioglioma_T1',  'Ganglioglioma_T1C+','Ganglioglioma_T2',
    'Germinoma_T1',      'Germinoma_T1C+',    'Germinoma_T2',
    'Glioblastoma_T1',   'Glioblastoma_T1C+', 'Glioblastoma_T2',
    'Granuloma_T1',      'Granuloma_T1C+',    'Granuloma_T2',
    'Medulloblastoma_T1','Medulloblastoma_T1C+','Medulloblastoma_T2',
    'Meningioma_T1',     'Meningioma_T1C+',   'Meningioma_T2',
    'Neurocytoma_T1',    'Neurocytoma_T1C+',  'Neurocytoma_T2',
    'No_Tumor_T1',       'No_Tumor_T2',
    'Oligodendroglioma_T1','Oligodendroglioma_T1C+','Oligodendroglioma_T2',
    'Papilloma_T1',      'Papilloma_T1C+',    'Papilloma_T2',
    'Schwannoma_T1',     'Schwannoma_T1C+',   'Schwannoma_T2',
    'Tuberculoma_T1',    'Tuberculoma_T1C+',  'Tuberculoma_T2',
]
print(f"✓ Loaded {len(CLASS_NAMES)} classes.")

# ------------------ HELPER FUNCTIONS ------------------
def create_gradient_background(canvas, width, height, color1, color2):
    """Create a smooth gradient background"""
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    
    for y in range(height):
        ratio = y / height
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(0, y, width, y, fill=color)

def create_rounded_button(parent, text, command, x, y, width, height, bg_color, fg_color, font_size):
    """Create a rounded button using canvas"""
    canvas = Canvas(parent, width=width, height=height, bg=SECONDARY_COLOR, highlightthickness=0, relief="flat")
    canvas.place(x=x, y=y)
    
    # Draw rounded rectangle
    radius = 15
    canvas.create_arc((0, 0, radius*2, radius*2), start=90, extent=90, fill=bg_color, outline=bg_color)
    canvas.create_arc((width-radius*2, 0, width, radius*2), start=0, extent=90, fill=bg_color, outline=bg_color)
    canvas.create_arc((0, height-radius*2, radius*2, height), start=180, extent=90, fill=bg_color, outline=bg_color)
    canvas.create_arc((width-radius*2, height-radius*2, width, height), start=270, extent=90, fill=bg_color, outline=bg_color)
    
    canvas.create_rectangle((radius, 0, width-radius, height), fill=bg_color, outline=bg_color)
    canvas.create_rectangle((0, radius, width, height-radius), fill=bg_color, outline=bg_color)
    
    # Add text
    text_id = canvas.create_text(width//2, height//2, text=text, fill=fg_color, font=("Arial", font_size, "bold"))
    
    # Hover effect
    def on_enter(event):
        canvas.delete("all")
        canvas.create_arc((0, 0, radius*2, radius*2), start=90, extent=90, fill=ACCENT_COLOR, outline=ACCENT_COLOR)
        canvas.create_arc((width-radius*2, 0, width, radius*2), start=0, extent=90, fill=ACCENT_COLOR, outline=ACCENT_COLOR)
        canvas.create_arc((0, height-radius*2, radius*2, height), start=180, extent=90, fill=ACCENT_COLOR, outline=ACCENT_COLOR)
        canvas.create_arc((width-radius*2, height-radius*2, width, height), start=270, extent=90, fill=ACCENT_COLOR, outline=ACCENT_COLOR)
        canvas.create_rectangle((radius, 0, width-radius, height), fill=ACCENT_COLOR, outline=ACCENT_COLOR)
        canvas.create_rectangle((0, radius, width, height-radius), fill=ACCENT_COLOR, outline=ACCENT_COLOR)
        canvas.create_text(width//2, height//2, text=text, fill=fg_color, font=("Arial", font_size, "bold"))
    
    def on_leave(event):
        canvas.delete("all")
        canvas.create_arc((0, 0, radius*2, radius*2), start=90, extent=90, fill=bg_color, outline=bg_color)
        canvas.create_arc((width-radius*2, 0, width, radius*2), start=0, extent=90, fill=bg_color, outline=bg_color)
        canvas.create_arc((0, height-radius*2, radius*2, height), start=180, extent=90, fill=bg_color, outline=bg_color)
        canvas.create_arc((width-radius*2, height-radius*2, width, height), start=270, extent=90, fill=bg_color, outline=bg_color)
        canvas.create_rectangle((radius, 0, width-radius, height), fill=bg_color, outline=bg_color)
        canvas.create_rectangle((0, radius, width, height-radius), fill=bg_color, outline=bg_color)
        canvas.create_text(width//2, height//2, text=text, fill=fg_color, font=("Arial", font_size, "bold"))
    
    def on_click(event):
        command()
    
    canvas.bind("<Enter>", on_enter)
    canvas.bind("<Leave>", on_leave)
    canvas.bind("<Button-1>", on_click)
    
    return canvas

def show_loading_animation(root):
    """Show animated loading indicator"""
    global loading_active
    loading_active = True
    
    def animate():
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while loading_active:
            result_label.config(text=f"{chars[i % len(chars)]} Analyzing MRI Image...", fg=ACCENT_COLOR)
            i += 1
            time.sleep(0.1)
    
    thread = threading.Thread(target=animate, daemon=True)
    thread.start()

# ------------------ IMAGE PREPROCESS ------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    return img

# ------------------ GUI CALLBACKS ------------------
def browse_image():
    global loading_active
    file_path = filedialog.askopenfilename(
        title="Select MRI Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        try:
            # Show loading animation
            show_loading_animation(root)
            
            image = Image.open(file_path)
            image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Add border effect to image
            img_with_border = Image.new('RGB', (320, 320), color=ACCENT_COLOR)
            offset = ((320 - image.width) // 2, (320 - image.height) // 2)
            img_with_border.paste(image, offset)
            
            photo = ImageTk.PhotoImage(img_with_border)
            image_label.config(image=photo, bg=SECONDARY_COLOR)
            image_label.image = photo
            
            root.after(500, lambda: predict_image_async(file_path))
        except Exception as e:
            loading_active = False
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

def predict_image_async(image_path):
    """Predict in background thread"""
    global loading_active
    
    def predict():
        try:
            img = preprocess_image(image_path)
            preds = model.predict(img, verbose=0)
            class_id = np.argmax(preds[0])
            confidence = preds[0][class_id] * 100
            
            # Simulate delay for effect
            time.sleep(0.5)
            
            loading_active = False
            display_result(CLASS_NAMES[class_id], confidence)
        except Exception as e:
            loading_active = False
            messagebox.showerror("Error", str(e))
    
    thread = threading.Thread(target=predict, daemon=True)
    thread.start()

def display_result(class_name, confidence):
    """Display prediction with color-coded confidence"""
    # Clean up class name for better display
    display_name = class_name.replace('_', ' ')
    
    if confidence >= 80:
        color = SUCCESS_COLOR
        confidence_text = "Very High Confidence"
    elif confidence >= 60:
        color = WARNING_COLOR
        confidence_text = "Good Confidence"
    else:
        color = ACCENT_COLOR
        confidence_text = "Moderate Confidence"
    
    result_text = f"🔬 {display_name}\n{confidence:.1f}% Confidence\n({confidence_text})"
    result_label.config(text=result_text, fg=color)

# ------------------ GUI SETUP ------------------
root = Tk()
root.title("🧠 Brain Tumor MRI Classifier")
root.geometry("700x850")
root.resizable(False, False)

# Set modern window background
root.config(bg=SECONDARY_COLOR)

# Create gradient background canvas
bg_canvas = Canvas(root, bg=SECONDARY_COLOR, highlightthickness=0)
bg_canvas.place(x=0, y=0, width=700, height=850)
create_gradient_background(bg_canvas, 700, 850, SECONDARY_COLOR, PRIMARY_COLOR)

# Main frame
main_frame = Frame(root, bg=SECONDARY_COLOR)
main_frame.pack(fill="both", expand=True)

# Title with glow effect
title_label = Label(
    main_frame, 
    text="🧠 Brain Tumor MRI Classifier",
    font=("Arial", 24, "bold"),
    bg=SECONDARY_COLOR,
    fg=ACCENT_COLOR,
    pady=20
)
title_label.pack()

# Subtitle
subtitle_label = Label(
    main_frame,
    text="AI-Powered Medical Image Analysis",
    font=("Arial", 11),
    bg=SECONDARY_COLOR,
    fg=LIGHT_COLOR,
    pady=5
)
subtitle_label.pack()

# Image display frame with shadow effect
image_frame = Frame(main_frame, bg=PRIMARY_COLOR, highlightthickness=2, highlightbackground=ACCENT_COLOR)
image_frame.pack(pady=20, padx=30)

image_label = Label(
    image_frame,
    bg=SECONDARY_COLOR,
    fg=TEXT_COLOR,
    width=20,
    height=10,
    text="📤 Select an MRI image to begin"
)
image_label.pack(padx=5, pady=5)

# Button frame
button_frame = Frame(main_frame, bg=SECONDARY_COLOR)
button_frame.pack(pady=20, padx=30, fill="x")

button_y = 520
create_rounded_button(
    root,
    text="📤 Select MRI Image",
    command=browse_image,
    x=125,
    y=button_y,
    width=450,
    height=55,
    bg_color=ACCENT_COLOR,
    fg_color=TEXT_COLOR,
    font_size=14
)

# Result display
result_label = Label(
    main_frame,
    text="Waiting for image selection...",
    font=("Arial", 12, "bold"),
    bg=SECONDARY_COLOR,
    fg=LIGHT_COLOR,
    pady=30,
    wraplength=600,
    justify="center"
)
result_label.pack(pady=30, padx=30)

# Footer
footer_label = Label(
    main_frame,
    text="✨ Powered by TensorFlow & EfficientNetV2 | 44 Tumor Classifications",
    font=("Arial", 9),
    bg=SECONDARY_COLOR,
    fg=LIGHT_COLOR,
    pady=10
)
footer_label.pack(side="bottom")

# Global variable for loading state
loading_active = False

root.mainloop()

# Handwritten Digit Recognition Model

An end-to-end project that trains a Convolutional Neural Network (CNN) from scratch on MNIST and serves a “draw-and-guess” web application where users can sketch digits and see live predictions.
---

What This System Does
The Handwritten Digit Recognition Model is a full-stack demonstration of how to train and deploy a deep learning model end-to-end:

Data Preparation & Training

Automatically downloads the MNIST dataset of 70,000 handwritten digit images.

Applies on-the-fly augmentations (small rotations and translations) to make the network robust to varied handwriting styles.

Trains a custom Convolutional Neural Network (CNN) from scratch for 10 epochs using PyTorch, saving the learned weights (digit_net.pth) for inference.

Real-Time Drawing Interface

Presents a 400×400 HTML5 <canvas> where users freely draw digits with mouse or touch.

Captures the drawing every 500 ms (and when the user lifts the pointer) and sends it to the server for prediction.

Image Preprocessing Pipeline

Receives the Base64-encoded PNG, decodes it via OpenCV, and composites any transparency over a white background.

Converts to grayscale, thresholds to isolate the digit, crops to its bounding box, pads to a square, and resizes to the 28×28 format the CNN expects.

Normalizes pixel values to match training conditions.

Model Inference & Feedback

Runs the preprocessed image through the trained CNN in Flask, obtaining a probability distribution over digits 0–9.

Splits the results into two side panels—digits 0–4 on the left, 5–9 on the right—highlighting the top guess.

Displays each probability as a percentage and as a horizontal progress bar for quick visual feedback.

---

## Repository Structure

```
digit-recognizer/
├── app/                   # Flask web application
│   ├── static/            # CSS & JavaScript assets
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── app.js
│   ├── templates/         # HTML templates
│   │   └── index.html
│   ├── __init__.py
│   ├── app.py             # Flask app entrypoint
│   ├── model.py           # CNN architecture (PyTorch)
│   └── preprocess.py      # Base64 → OpenCV → PyTorch tensor
│
├── data/                  # MNIST data and any collected feedback
│
├── models/                # Checkpoints
│   └── digit_net.pth      # Trained model weights
│
├── training/              # Training scripts
│   ├── __init__.py
│   ├── dataset.py         # Data loader with augmentations
│   └── train.py           # Train script (10 epochs by default)
│
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/Khayal-Aghazada/handwritten-digit-recognition-model
cd digit-recognizer
pip install -r requirements.txt
```

**requirements.txt** includes:
```
torch
torchvision
flask
opencv-python
numpy
tqdm
```

---

### 2. Train the Model

The training script will download MNIST (if needed), apply simple augmentations, and train (or continue training) for 10 epochs, saving the best model to `models/digit_net.pth`.

```bash
cd training
python train.py
```

- Downloads MNIST into `data/`  
- Applies random rotations and translations to training data  
- Trains for 10 epochs (CPU or GPU)  
- Saves the model weights to `models/digit_net.pth`

---

### 3. Run the Web App

In a separate terminal, start the Flask server:

```bash
cd app
python app.py
```

or from project root:

```bash
python -m app.app
```

- Opens a server at **http://127.0.0.1:5000**  
- Provides a 400×400 canvas where you can draw a digit  
- Shows live-updating predictions for digits **0–4** (left panel) and **5–9** (right panel)  
- Displays horizontal progress bars indicating prediction confidence  

---

## How It Works

1. **Frontend**  
   - HTML5 `<canvas>` for drawing  
   - JavaScript captures strokes, sends the image every 500 ms and on pen-up to `/predict`  
   - Results rendered as percentages and bars in side panels  

2. **Backend**  
   - Flask serves the web page and handles `/predict` requests  
   - `preprocess.py` uses OpenCV to decode, threshold, crop, center, and resize the drawing to 28×28  
   - PyTorch loads `digit_net.pth` and runs inference, returning probabilities for each digit  

3. **Model**  
   - A simple CNN defined in `model.py`  
   - Trained from scratch on MNIST with data augmentations  
   - Saved as `models/digit_net.pth`

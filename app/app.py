import os
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F

from model import SimpleCNN
from preprocess import preprocess_image

# Compute project root (one level above this file’s folder)
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'digit_net.pth')

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static')
)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_b64 = request.json['image']
    tensor  = preprocess_image(img_b64)
    tensor  = torch.from_numpy(tensor).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    return jsonify({str(i): float(probs[i]) for i in range(10)})

if __name__ == '__main__':
    # Always run from project root: `python -m app.app`
    app.run(debug=True)

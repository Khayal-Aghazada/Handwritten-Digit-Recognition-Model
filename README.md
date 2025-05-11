# Handwritten Digit Recognition Model

An end-to-end Python project that trains a Convolutional Neural Network (CNN) from scratch on MNIST and serves a live “draw-and-guess” web interface where users can sketch digits and see real-time predictions—no external model downloads or paid APIs required.

🚀 **Features**
- **Live Drawing Interface**  
  400×400 HTML5 `<canvas>` where users draw digits with mouse or touch  
- **Real-Time Guessing**  
  Throttled server calls every 500 ms + on pointer-up to update predictions instantly  
- **Dual-Panel Display**  
  Digits **0–4** in the left sidebar, **5–9** in the right sidebar, each with percentage and progress bar  
- **Custom Preprocessing**  
  OpenCV pipeline: Base64 → composite α → grayscale → threshold → crop → center → resize (28×28)  
- **Train From Scratch**  
  PyTorch CNN trained with MNIST + simple augmentations (rotation, translation)

📦 **Technology Stack**
- **Language & Frameworks**: Python 3.8+, Flask, PyTorch, torchvision, OpenCV  
- **Frontend**: HTML5 Canvas, CSS3 (Flexbox, gradients), Vanilla JavaScript (Fetch API)  
- **Data**: MNIST dataset (auto-downloaded)  
- **Training**: `training/train.py` script, uses `training/dataset.py` for data loading & augmentation  

🔧 **Installation & Setup**
```bash
git clone https://github.com/Khayal-Aghazada/handwritten-digit-recognition-model.git
cd handwritten-digit-recognition-model
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

▶️ **Train the Model**
```bash
cd training
python train.py
```
- Downloads MNIST into `../data/`  
- Applies random rotations (±15°) and translations (±10%)  
- Trains (or fine-tunes) for 10 epochs  
- Saves best and latest weights to `../models/digit_net.pth`

▶️ **Run the Web App**
```bash
# From project root
python -m app.app
```
or
```bash
cd app
python app.py
```
- Launches Flask server at **http://127.0.0.1:5000**  
- Draw a digit, see live predictions with confidence bars

📁 **Project Structure**
```
handwritten-digit-recognition-model/
├── app/                       # Flask web application
│   ├── static/                # CSS & JS assets
│   │   ├── css/style.css
│   │   └── js/app.js
│   ├── templates/index.html   # Drawing UI
│   ├── app.py                 # Flask routes
│   ├── model.py               # CNN definition
│   └── preprocess.py          # OpenCV preprocessing
│
├── data/                      # MNIST dataset & feedback (optional)
├── models/                    # Model checkpoints
│   └── digit_net.pth
├── requirements.txt           # Python dependencies
├── training/                  # Training scripts
│   ├── dataset.py             # DataLoader & transforms
│   └── train.py               # 10-epoch training script
└── README.md                  # This file
```

🛠️ **Customization**
- **Canvas Size & Style**: Edit `templates/index.html` and `static/css/style.css`  
- **Prediction Interval**: Adjust `GUESS_INTERVAL` in `app/static/js/app.js`  
- **Model Architecture**: Modify `app/model.py` for deeper or wider CNN  
- **Augmentations**: Tweak rotation/translation ranges in `training/dataset.py`

🤝 **Contributing**
1. Fork this repository  
2. Create a feature branch (`git checkout -b feature/xyz`)  
3. Make your changes and commit (`git commit -m "feat: Add xyz"`)  
4. Push to your branch and open a Pull Request  

📜 **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

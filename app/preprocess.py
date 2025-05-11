import cv2
import numpy as np
import base64

def preprocess_image(img_b64: str) -> np.ndarray:
    # 1) Decode base64 to image (may include alpha)
    header, data = img_b64.split(',', 1)
    img_data = base64.b64decode(data)
    arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    # 2) Composite alpha over white if needed, then convert to grayscale
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        rgb = img[:, :, :3]
        white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
        img = (rgb * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # else: already grayscale

    # 3) Threshold and invert: digit → white (255), BG → black (0)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 4) Crop to bounding box of the digit
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]

    # 5) Square-pad to maintain aspect ratio
    m = max(w, h)
    square = np.zeros((m, m), dtype=np.uint8)
    y_off = (m - h) // 2
    x_off = (m - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    # 6) Resize to 28×28 and normalize
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0

    # 7) Add batch and channel dimensions: [1,1,28,28]
    return norm.reshape(1, 1, 28, 28)

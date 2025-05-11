# training/train.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Ensure `app/` is on the import path so we can load model.py
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from dataset import get_dataloaders       # your training/dataset.py
from app.model   import SimpleCNN         # the CNN in app/model.py

# Paths
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'digit_net.pth')

def train_one_epoch(model, device, loader, optimizer, criterion, epoch):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch}")
    running_loss = 0.0

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader.dataset)
    print(f" → Train Loss: {avg_loss:.4f}")

def evaluate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            running_loss += criterion(outputs, labels).item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset) * 100
    print(f" → Val   Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    extra_epochs = 1
    batch_size   = 64
    lr           = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Model
    model = SimpleCNN().to(device)
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # 2) Data
    train_loader, test_loader = get_dataloaders(batch_size=batch_size, data_dir=DATA_DIR)

    # 3) Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Ensure models/ exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, extra_epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, criterion, epoch)
        acc = evaluate(model, device, test_loader, criterion)

        # Save checkpoint
        torch.save(model.state_dict(), MODEL_PATH)
        if acc > best_acc:
            best_acc = acc
            print(f" ★ New best accuracy: {best_acc:.2f}%\n")

    print(f"Finished {extra_epochs} epochs. Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()

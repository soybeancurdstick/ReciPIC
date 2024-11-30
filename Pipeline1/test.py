import torch
import os
import matplotlib.pyplot as plt
from model import ResNet, resnet50_config
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

# Directories and device setup
ROOT = '/Users/kaylynl/Desktop/ReciPIC/data'
data_dir = os.path.join(ROOT, 'data3')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the number of output classes
output_dim = 36  # Automatically get the number of classes

# Initialize the model with the correct output_dim
model = ResNet(resnet50_config, output_dim=output_dim).to(device)

# Training and validation transforms
pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(pretrained_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

test_transforms = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.CenterCrop(pretrained_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

# Datasets and loaders
train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation functions
def train(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return epoch_loss / len(loader), accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return epoch_loss / len(loader), accuracy

# Training loop
EPOCHS = 10
BEST_VALID_LOSS = float('inf')

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

    # Save the best model
    if valid_loss < BEST_VALID_LOSS:
        BEST_VALID_LOSS = valid_loss
        torch.save(model.state_dict(), "model.pth")
        print("Saved Best Model!")

# Load the best model and test
model.load_state_dict(torch.load("model.pth"))

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

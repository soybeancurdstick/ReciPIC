import torch
import os
from binary_model import load_binary_model, classify_food, train_binary_model
from ingredient_model import load_ingredient_model, train_ingredient_model
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from time import time


# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load both models
binary_model = load_binary_model(device)
ingredient_model = load_ingredient_model(device)

# Define transformations for input images (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets (for training and evaluation)
trainset = ImageFolder(root='./binary_data3/train', transform=transform)
valset = ImageFolder(root='./binary_data3/val', transform=transform)
testset = ImageFolder(root='./binary_data3/test', transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


import torch
import torch.nn.functional as F

def predict_ingredient(model, image, device):
    """
    Predict the ingredient for an image using the ingredient classification model.

    Args:
        model (torch.nn.Module): The trained ingredient classification model.
        image (torch.Tensor): The input image tensor.
        device (torch.device): The device (CPU or GPU) to run the model on.

    Returns:
        tuple: Predicted class (ingredient name) and confidence score (softmax probability).
    """
    model.eval()  # Set the model to evaluation mode
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        outputs = model(image)  # Forward pass through the model
        probs = F.softmax(outputs, dim=1)  # Get softmax probabilities
        confidence, predicted_class = torch.max(probs, 1)  # Get the predicted class and confidence score

    # Get the predicted ingredient class label (you can map class indices to ingredient names)
    ingredient_class_index = predicted_class.item()
    ingredient_class_name = model.classes[ingredient_class_index]  # Assuming 'classes' is an attribute of the model

    return ingredient_class_name, confidence.item()


# Training the binary model
def train_combined_model(binary_model, ingredient_model, trainloader, valloader, device, num_epochs=20):
    # Train binary classification model
    print("Training Binary Classification Model (Food vs other)...")
    train_binary_model(binary_model, trainloader, valloader, device, num_epochs)

    # After training the binary model, we train the ingredient model
    print("Training Ingredient Classification Model...")
    train_ingredient_model(ingredient_model, trainloader, valloader, device, num_epochs)

# Train the models
train_combined_model(binary_model, ingredient_model, trainloader, valloader, device, num_epochs=20)

# Evaluate the performance after training
def evaluate_combined_model(binary_model, ingredient_model, data_loader, device):
    y_true = []  # True labels for food classification
    y_pred_binary = []  # Predicted labels for binary classification
    y_pred_ingredient = []  # Predicted ingredient classes

    # For binary classification (food vs. not food)
    ingredient_predictions = []  # To store the ingredient predictions (if food)
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        
        # Classify as food or not food
        binary_prediction = classify_food(binary_model, inputs, device)
        
        if binary_prediction:  # If classified as 'food'
            # Predict the ingredient
            predicted_class, confidence_score = predict_ingredient(ingredient_model, inputs, device)
            ingredient_predictions.append((predicted_class, confidence_score))
        else:
            ingredient_predictions.append((None, None))  # No ingredient for non-food

        y_true.extend(labels.cpu().numpy())
        y_pred_binary.extend([binary_prediction] * len(labels))  # Binary predictions (food or not food)

    # Calculate the overall performance
    binary_accuracy = accuracy_score(y_true, y_pred_binary)
    print(f'Binary Classification (Food vs Not Food) Accuracy: {binary_accuracy * 100:.2f}%')

    # Collect the ingredient-level classification (only for images classified as food)
    ingredient_true = [label for label, pred in zip(y_true, ingredient_predictions) if pred[0] is not None]
    ingredient_pred = [pred[0] for pred in ingredient_predictions if pred[0] is not None]
    
    ingredient_accuracy = accuracy_score(ingredient_true, ingredient_pred)
    print(f'Ingredient Classification Accuracy: {ingredient_accuracy * 100:.2f}%')

    # Print a more detailed classification report (if needed)
    print("Classification report for ingredient predictions (if food):")
    print(classification_report(ingredient_true, ingredient_pred))

# Run evaluation after training
evaluate_combined_model(binary_model, ingredient_model, testloader, device)

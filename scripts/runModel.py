import sys
import os

# Add the project directory to the Python path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from PIL import Image
import torch
from torchvision import transforms
from ultralytics import YOLO
from Pipeline1.model import ResNet, resnet50_config

# Load YOLO model
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'yolov8n.pt')
yolo_model = YOLO(YOLO_MODEL_PATH)

# Load ResNet model
RESNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'resnet.pth')
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 36)  # Adjust number of classes
resnet.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=torch.device('cpu')))
resnet.eval()

# Define image transformations for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
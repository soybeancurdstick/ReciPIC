import sys
import os
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from bs4 import BeautifulSoup
import requests

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocessing function for ResNet-50
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization values for ImageNet
])

# Load pre-trained ResNet-50 model
print("Loading ResNet-50 model...")
try:
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(os.listdir('./processed_img')))  # Adjust for your dataset classes
    model.load_state_dict(torch.load('resnet50_params.pt'))  # Ensure this path is correct
    model.eval()
    print("ResNet-50 model loaded successfully.")
except Exception as e:
    print(f"Error loading ResNet-50 model: {e}")
    model = None

# Load YOLO model
print("Loading YOLO model...")
try:
    yolo_model = YOLO('yolov8n.pt')  # Ensure this path is correct
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

# Custom dataset for class labels
try:
    custom_dataset = ImageFolder(root='./processed_img', transform=preprocess)
    class_labels = custom_dataset.classes
except Exception as e:
    print(f"Error loading dataset: {e}")
    custom_dataset = None
    class_labels = []

# Scrape recipes
def scrape_recipes(ingredients):
    urls = []
    for ingredient in ingredients:
        search_url = f"https://www.allrecipes.com/search/results/?wt={ingredient}"
        try:
            response = requests.get(search_url)
            if response.ok:
                soup = BeautifulSoup(response.text, 'html.parser')
                urls += [link['href'] for link in soup.find_all('a', class_='card__titleLink', href=True)]
        except Exception as e:
            print(f"Error scraping recipes for {ingredient}: {e}")
    return urls[:5]

# Classify image
def classify_image(cropped_image):
    try:
        cropped_image = cropped_image.convert('RGB')
        input_tensor = preprocess(cropped_image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        _, index = torch.max(output, 1)
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        predicted_class = class_labels[index[0]]
        confidence = percentage[index[0]].item()
        return predicted_class, confidence
    except Exception as e:
        print(f"Error classifying image: {e}")
        return None, None

@app.route('/home', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Process the image
            image = Image.open(filepath)
            results = yolo_model(image)
            ingredients = []

            for result in results:
                for bbox in result.boxes:
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0].tolist())
                    cropped_image = image.crop((x1, y1, x2, y2))
                    predicted_class, _ = classify_image(cropped_image)
                    if predicted_class:
                        ingredients.append(predicted_class)

            # Get recipe URLs
            recipe_urls = scrape_recipes(ingredients)
            return render_template('index.html', recipe_urls=recipe_urls, ingredients=ingredients)

        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template('index.html', error="An error occurred while processing the image.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)

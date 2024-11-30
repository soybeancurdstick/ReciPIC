import sys
import os

# Add the project directory to the Python path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

import importlib.util
from bs4 import BeautifulSoup
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import requests
import torch
from Pipeline1.model import ResNet, resnet50_config


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC




# Print the current working directory and path details
print("Current Working Directory:", os.getcwd())

# Define the path to your model module
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipeline1', 'model.py'))
print("Path to module:", module_path)

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocessing function for ResNet
preprocess_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom dataset for class labels
processed_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_img'))

if os.path.exists(processed_img_path):
    try:
        custom_dataset = ImageFolder(root=processed_img_path, transform=preprocess_resnet)
        class_labels = custom_dataset.classes
    except Exception as e:
        print(f"Error loading dataset: {e}")
        custom_dataset = None
        class_labels = []
else:
    print(f"Dataset directory not found: {processed_img_path}")
    custom_dataset = None
    class_labels = []

# Load models once, globally
print("Loading models...")
output_dim = 36  # Matches the checkpoint
#init_params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'init_params.pt'))
tut5_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tut5-model.pt'))


try:
    #if not os.path.exists(init_params_path):
    if not os.path.exists(tut5_path):
        #raise FileNotFoundError(f"init_params.pt not found at {init_params_path}")
        raise FileNotFoundError(f"init_params.pt not found at {tut5_path}")

    model = ResNet(resnet50_config, output_dim)
    #state_dict = torch.load(init_params_path, map_location=torch.device('cpu'))
    state_dict = torch.load(tut5_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    yolo_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n.pt'))
    yolo_model = YOLO(yolo_model_path)

    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    yolo_model = None

# Define food and non-food classes
food_classes = ['apple', 'banana', 'bellpepper', 'bread', 'broccoli', 'carrot', 'celery', 'cinnamon', 'corn', 'cucumber', 'egg', 'garlic', 'kale', 'lemon', 'lettuce', 'lime', 'mango', 'mushroom', 'onion', 'orange', 'potato', 'salt shaker', 'tomato', 'tomato sauce']
other_classes = ['bottle', 'bowl', 'chair', 'cup', 'dining table', 'fork', 'jar', 'bed', 'bicycle', 'bottle', 'bowl', 'cup', 'dining table', 'knife', 'glass', 'person', 'potted plant', 'spoon', 'vase', 'wine glass']

# Scrape recipes from Food.com

def scrape_recipes(ingredients):
    ingredient_query = "+".join(ingredients)
    search_url = f"https://www.food.com/search/{ingredient_query}"
    print(f"Searching for recipes: {search_url}")

    # Initialize the WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")  # Optional: Maximize the window
    # chrome_options.add_argument("--headless")  # Uncomment if you want headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        driver.get(search_url)

        # Explicit wait for recipe links to be loaded
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "card__titleLink"))
        )

        # Find recipe links
        urls = []
        recipe_links = driver.find_elements(By.CLASS_NAME, "card__titleLink")
        for link in recipe_links:
            urls.append(link.get_attribute("href"))

        return urls[:5]

    except Exception as e:
        print(f"Error scraping recipes: {e}")
    finally:
        driver.quit()

    return []


# Example usage
#ingredients = ["banana", "egg"]
#recipe_urls = scrape_recipes(ingredients)
#print("Found recipe URLs:", recipe_urls)

# Classify image using ResNet
def classify_image(cropped_image):
    try:
        cropped_image = cropped_image.convert('RGB')
        input_tensor = preprocess_resnet(cropped_image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        if isinstance(output, tuple):  # Check if the output is a tuple (e.g., from a model returning logits and embeddings)
            output = output[0]
        _, index = torch.max(output, dim=1)  # Use `dim=1` to specify the axis for max calculation
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        predicted_class = class_labels[index.item()]
        confidence = percentage[index.item()].item()
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
            # Process the image with YOLO
            image = Image.open(filepath)
            results = yolo_model(image)
            ingredients = []

            # Print detected classes to debug
            print("Detected classes from YOLO:", results)

            for result in results:
                for bbox in result.boxes:
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0].tolist())
                    cropped_image = image.crop((x1, y1, x2, y2))

                    # Classify the cropped image with ResNet
                    predicted_class, confidence = classify_image(cropped_image)

                    # If the predicted class is a food-related class, add it to ingredients
                    if (
                        predicted_class in food_classes
                        and predicted_class not in other_classes
                        and predicted_class not in ingredients
                     ):
                        ingredients.append(predicted_class)

            # Get recipe URLs based on food ingredients
            print("Detected food ingredients:", ingredients)

            if ingredients:
                recipe_urls = scrape_recipes(ingredients)
                print("Found recipe URLs:", recipe_urls)
            else:
                recipe_urls = []

            return render_template('index.html', recipe_urls=recipe_urls, ingredients=ingredients)

        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template('index.html', error="An error occurred while processing the image.")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
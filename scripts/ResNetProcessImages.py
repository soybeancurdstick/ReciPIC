import os
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

print("ResNet for classification")

# 1. Load YOLOv8 model for object detection
yolo_model_path = os.path.join(os.getcwd(), '..', 'models', 'yolov8n.pt')  # Corrected path to yolov8n.pt
yolo_model = YOLO(yolo_model_path)

# 2. Load a pre-trained ResNet model for image classification
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# 3. Load ImageNet class names for ResNet classification
imagenet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ImageNet1000Classes.txt')   # Corrected path for ImageNet classes
with open(imagenet_path) as classfile:
    ImageNetClasses = [line.strip() for line in classfile.readlines()]

# 4. Define preprocessing transformations for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to classify the cropped image using ResNet
def classify_image(cropped_image):
    # Convert the image to RGB (in case it's not)
    cropped_image = cropped_image.convert('RGB')

    # Preprocess the cropped image
    input_tensor = preprocess(cropped_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Pass the cropped image through ResNet
    with torch.no_grad():
        output = resnet_model(input_batch)

    # Get the predicted class index and the confidence score
    _, index = torch.max(output, 1)
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    predicted_class = ImageNetClasses[index[0]]
    confidence = percentage[index[0]].item()

    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")
    return predicted_class, confidence

# Function to check if the file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# 5. Crop, classify, and save detected objects
def crop_save_and_classify(image_path, detections, output_dir):
    image = Image.open(image_path)

    for i, box in enumerate(detections.boxes.xyxy):
        box = box.cpu().numpy()
        class_index = int(detections.boxes.cls[i])
        label = yolo_model.names[class_index]

        # Ensure box coordinates are in pixel values
        x1, y1, x2, y2 = [int(coord) for coord in box]

        # Crop detected object
        cropped_img = image.crop((x1, y1, x2, y2))

        # Convert the cropped image to 'RGB' mode if it's in 'RGBA'
        if cropped_img.mode == 'RGBA':
            cropped_img = cropped_img.convert('RGB')

        # Create output directory for the specific class (e.g., 'BellPepper')
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)  # Ensure the directory exists

        # Save the cropped image with a unique name
        cropped_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{label}_{i}.jpg"
        output_path = os.path.join(label_dir, cropped_filename)

        # Save the cropped image
        cropped_img.save(output_path)
        print(f"Saved cropped image to: {output_path}")

        # Classify the cropped image using ResNet
        predicted_class, confidence = classify_image(cropped_img)
        print(f"Detected '{label}' classified as '{predicted_class}' with {confidence:.2f}% confidence.")

# 6. Main processing loop for the directory of images
input_dir = os.path.join(os.getcwd(), 'data', 'unprocessed_img')  # Corrected path to 'data/unprocessed_img'
output_dir = os.path.join(os.getcwd(), 'data', 'processed_img1')  # Corrected path to 'data/processed_img1'
os.makedirs(output_dir, exist_ok=True)

for image_filename in os.listdir(input_dir):
    if is_image_file(image_filename):  # Only process image files
        image_path = os.path.join(input_dir, image_filename)
        img = cv2.imread(image_path)

        # YOLOv8 detection
        results = yolo_model(img)
        print(f"Detected objects in {image_filename}: {results[0]}")

        # Crop, save, and classify using ResNet
        crop_save_and_classify(image_path, results[0], output_dir)
    else:
        print(f"Skipping non-image file: {image_filename}")

print("Image processing and classification completed.")

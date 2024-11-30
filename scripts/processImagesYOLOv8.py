import os
from ultralytics import YOLO
from PIL import Image
import cv2


print("YOLOv8 for classification");
# Load YOLOv8 model
model = YOLO('./models/yolov8n.pt')
print(model.names)

input_dir = './data/unprocessed_img'
output_dir = './data/processed_img1'
os.makedirs(output_dir, exist_ok=True)

# Function to check if the file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# Crop and save detected images
def crop_save(image_path, detections, output_dir):
    image = Image.open(image_path)

    for detection in detections:
        for i, box in enumerate(detection.boxes.xyxy):
            box = box.cpu().numpy()
            class_index = int(detection.boxes.cls[i])
            label = model.names[class_index]

            # Ensure box coordinates are in pixel values
            x1, y1, x2, y2 = [int(coord) for coord in box]

            # Crop detected object
            cropped_img = image.crop((x1, y1, x2, y2))

             # Convert the cropped image to 'RGB' mode if it's in 'RGBA'
            if cropped_img.mode == 'RGBA':
                cropped_img = cropped_img.convert('RGB')

            # Create output directory for specific class if it doesn't exist
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            cropped_filename = f"{os.path.splitext(image_filename)[0]}_{label}_{i}.jpg"
            output_path = os.path.join(label_dir, cropped_filename)

            # Save the cropped image
            cropped_img.save(output_path)

# Loop through images in folder
for image_filename in os.listdir(input_dir):
    if is_image_file(image_filename):  # Only process image files
        image_path = os.path.join(input_dir, image_filename)
        img = cv2.imread(image_path)

        # YOLOv8 detection
        results = model(img)
        print(results[0])

        crop_save(image_path, results, output_dir)
    else:
        print(f"Skipping non-image file: {image_filename}")  # Print a message for skipped files

print("Image processing completed.")

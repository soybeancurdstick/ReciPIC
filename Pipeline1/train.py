import os
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet, resnet50_config
import copy

# Set the seed
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths and directories
ROOT = 'Csci487'
data_dir = os.path.join(ROOT, 'data3')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

######SPLIT DATA INTO TEST AND TRAIN
# Define the train/test split ratio
TRAIN_RATIO = 0.8

# Specify the root directory where your images are stored
src_dir = '../processed_img'
destination_dir = 'data3'
train_dir = os.path.join(destination_dir, 'train')
test_dir = os.path.join(destination_dir, 'test')

# Remove existing train/test directories if they exist
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

# Create new train/test directories
os.makedirs(train_dir)
os.makedirs(test_dir)

# Get the class names (apple, banana, etc.)
classes = os.listdir(src_dir)
print(classes)
# Loop through each class directory (e.g., apple, banana)
for c in classes:
    class_dir = os.path.join(src_dir, c)

    # Skip train and test directories to avoid them being processed as classes
    if c in ['train', 'test', '.DS_Store']:
        continue

    # Get all images in the class directory
    images = os.listdir(class_dir)

    # Split the images into training and testing sets
    n_train = int(len(images) * TRAIN_RATIO)
    train_images = images[:n_train]
    test_images = images[n_train:]

    # Create subdirectories for the class in train and test directories
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(test_dir, c), exist_ok=True)

    # Copy the training images to the train directory
    for image in train_images:
        image_src = os.path.join(class_dir, image)
        image_dst = os.path.join(train_dir, c, image)
        shutil.copyfile(image_src, image_dst)

    # Copy the testing images to the test directory
    for image in test_images:
        image_src = os.path.join(class_dir, image)
        image_dst = os.path.join(test_dir, c, image)
        shutil.copyfile(image_src, image_dst)
print(f"Train directory: {train_dir}")
print(f"Does train directory exist? {os.path.exists(train_dir)}")
print("Data split into training and testing sets in 'data2' folder.")



###CAL MEAN AND STD TO NORMALIZE DATA SET
#load train data from train folder
train_data = datasets.ImageFolder(root = train_dir, 
                                  transform = transforms.ToTensor())

#3 for (red, green, blue)
means = torch.zeros(3)
stds = torch.zeros(3)

for img, label in train_data:
    means += torch.mean(img, dim = (1,2))
    stds += torch.std(img, dim = (1,2))

means /= len(train_data)
stds /= len(train_data)
    
print(f'Calculated means: {means}')
print(f'Calculated stds: {stds}')

#LOAD DATA
pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(pretrained_size, padding = 10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.CenterCrop(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

#LOAD DATA wITH TRANSFORMER
train_data = datasets.ImageFolder(root = train_dir, 
                                  transform = train_transforms)

test_data = datasets.ImageFolder(root = test_dir, 
                                 transform = test_transforms)
print("Loaded data with transformers")


#Validation split
VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data, 
                                           [n_train_examples, n_valid_examples])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms


print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


#Iterators with larbe batch size
BATCH_SIZE = 64

train_iterator = data.DataLoader(train_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data, 
                                 batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(test_data, 
                                batch_size = BATCH_SIZE)


#Check if images were processed correctly
#renormalize images so color looks right
print("Normalizing and sampling images")
def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image 

def plot_images(images, labels, classes, normalize = True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis('off')

N_IMAGES = 25

images, labels = zip(*[(image, label) for image, label in 
                           [train_data[i] for i in range(N_IMAGES)]])

classes = test_data.classes

plot_images(images, labels, classes)

EPOCHS = 30
STEPS_PER_EPOCH = len(train_iterator)
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
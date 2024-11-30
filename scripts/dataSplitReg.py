import os
import shutil
import splitfolders

print("For binary model (food and other)")
# Paths for input and output directories
input_folder = './data/processed_img'  # Original dataset with all classes
output_folder = './data/binary_img'   # Folder for binary classification (food vs. other)

# Step 1: Split the dataset into train, val, and test using splitfolders
splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(.8, .1, .1))

# Define the class categories for food and other
food_classes = ['apple', 'banana', 'bellpepper', 'bread','broccoli', 'carrot','celery','cinnamon','corn','cucumber','egg','garlic','kale', 'lemon','lettuce', 'lime', 'mango', 'mushroom','onion','orange', 'potato','salt shaker', 'tomato', 'tomato sauce']
other_classes = ['bottle','bowl','car', 'chair','cup','dining table','fork','jar','bed', 'bicycle', 'bottle','bowl','cup', 'dining table','knife', 'glass', 'person', 'potted plant','spoon', 'vase', 'wine glass']  # Add more non-food-related classes

# Step 2: Create "food" and "other" directories in train, val, and test folders
def organize_binary_folders(base_dir):
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        
        # Create 'food' and 'other' directories in each split
        food_dir = os.path.join(split_dir, 'food')
        other_dir = os.path.join(split_dir, 'other')
        os.makedirs(food_dir, exist_ok=True)
        os.makedirs(other_dir, exist_ok=True)
        
        # Iterate through class folders in the current split directory
        for class_folder in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_folder)
            
            # Skip if it's the newly created 'food' or 'other' folders
            if class_folder in ['food', 'other']:
                continue
            
            # Determine whether the class belongs to food or other
            if class_folder in food_classes:
                target_dir = food_dir
            elif class_folder in other_classes:
                target_dir = other_dir
            else:
                continue  # Skip classes not listed in either category
            
            # Move all files from class folder to the respective 'food' or 'other' folder
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    src_file = os.path.join(class_path, file_name)
                    dst_file = os.path.join(target_dir, file_name)
                    shutil.move(src_file, dst_file)
                
                # Remove the now-empty class folder
                os.rmdir(class_path)

# Step 3: Organize the data into 'food' and 'other' folders
organize_binary_folders(output_folder)

print("Dataset successfully organized for binary classification (food vs. other).")

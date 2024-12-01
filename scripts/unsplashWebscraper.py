import os
import requests

# Function to download an image from a URL
def download_image(image_url, save_dir, img_name):
    try:
        # Get the image response
        img_data = requests.get(image_url).content
        # Create the directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Write the image to the specified file
        with open(os.path.join(save_dir, img_name), 'wb') as handler:
            handler.write(img_data)
        print(f"Downloaded: {img_name}")
    except Exception as e:
        print(f"Error downloading {img_name}: {e}")

# Function to fetch images from Unsplash API
def fetch_food_images(query, save_dir, num_images):
    # Set your API key here
    API_KEY = 'jKcsYL30VujCMkPO8fxVAlbIPi_L0q3N7UcSJSgBbpU'  # Replace with your Unsplash Access Key
    headers = {
        'Authorization': f'Client-ID {API_KEY}'
    }

    # URL for searching images
    url = f'https://api.unsplash.com/search/photos?query={query}&per_page={num_images}'

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        for i, photo in enumerate(data['results']):
            img_url = photo['urls']['regular']  # Get the regular image URL
            img_name = f"{query}_{i + 1}.jpg"  # Create a unique name for each image
            download_image(img_url, save_dir, img_name)  # Download the image
    else:
        print(f"Failed to retrieve data from Unsplash. Status code: {response.status_code}")

# Example usage
if __name__ == "__main__":
    reciPIC_dir = reciPIC_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ReciPIC'))


    print("dir : " + reciPIC_dir)
    
    food_item = input('Enter ingredient you want to web scrape: ')  # Change this to the food item you want
    save_directory = os.path.join(reciPIC_dir, 'data', 'unprocessed_img') 
    num_img = int(input('Enter number of images you want to scrape: '))
    fetch_food_images(food_item, save_directory, num_img)

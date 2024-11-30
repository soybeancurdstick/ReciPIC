ReciPIC is a proof of concept project that demonstrates the potential of using machine learning to recognize food ingredients from images and suggest recipes based on detected items. This manual provides a guide to the software's functionality and usage.


Features
Detects food ingredients from images using a custom YOLOv8 model.
Retrieves recipes from the web using BeautifulSoup and Selenium.
Dynamic file handling and flexible input methods (drag-and-drop, file upload).
Web interface built with Flask to interact with the system.


ReciPIC/
├── models/             # Pretrained and fine-tuned YOLOv8 and Resnet models
├── static/             # Static files for the web interface (CSS, JS, images)
├── templates/          # HTML templates for Flask
├── data/               # Dataset used for training and testing
├── Pipeline1/          # ResNet model and associated scripts
├── scripts/            # Web scraping and utility scripts
├── app.py              # Flask application entry point
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation


Technologies Used
Programming Languages: Python
Frameworks:
Flask for the web application
Machine Learning:
YOLOv8 for object detection
ResNet for image classification
Web Scraping:
BeautifulSoup
Selenium
Image Processing: PIL (Pillow), OpenCV
Other Tools:
Torch, torchvision
ChromeDriverManager for Selenium setup


Setup
Prerequisites
Python 3.8 or later
Pip and virtual environment tools (recommended)


Installation Steps

1. Clone the Repository:
	git clone https://github.com/soybeancurdstick/ReciPIC.git

2. Open terminal and go to project root folder (reciPIC)

3. Set Up a Virtual Environment:
	python3 -m venv env

4. Install Dependencies:
	pip install -r requirements.txt

5. Change directory to ReciPIC
	cd ReciPic


To run app:
1. Change directory to app
	cd app

2. Run the Application:
	python3 modelConnect.py

3. Access the Application:
	Open your browser and go to http://127.0.0.1:5000/home.



To webscrape:
1. Change directory to script
	cd scripts

2. Run the unsplashWebscraper:
	python3 unsplashWebscraper.py

3. Enter ingredient you want to webscrape in terminal prompt (example below uses potato):
	Enter ingredient you want to web scrape: potato

4. Web Scraped images will be saved in ReciPIC/data/unprocessed_img folder.



To crop and label data:
1. Change directory to script
	cd scripts
2. Run ResNetProcessImages.
	python3 ResNetProcessImages.py
3. Processed images will be saved in ReciPIC/data/processed_img1 folder.
4. Manually correct labels and put final images in processed_img (ReciPIC/data/processed_img).


To train and test model:
1. Change directory to Pipeline1
	cd Pipeline1
2. Run trainTest.
	python3 trainTest.py
3. Enter number of epochs (example below uses 30):
	Enter epochs: 30



Usage
1. Upload an image of food ingredients through the web interface.
2. The system detects the ingredients and displays a list.
3. View suggested recipes scraped from websites like Food.com.
This project is focused on image recognition of food ingredients. 

Folder descriptions:

Binary_data folders: holds split data for training
Processed_img: holds images that have been processed and label
Unprocessed_img: holds images that have not been processed
used_img: holds images that I've used.

ImageNet1000Classes.txt: 1000 class text (ResNet)

.py files:

processImagesYOLOv8.py: gets images from unprocess_img folder and crops and labels using YOLOv8, saving labeled images to proper folder in processed_img or creating new folder for new label.

ResNetProcessImages.py: gets images from unprocess_img folder and crops using YOLOv8 and labels using ResNet, saving labeled images to proper folder in processed_img or creating new folder for new label.

webScrapeData.py: scrapes user specified data from Pexels website using pixel api. 

splitDataLabels.py: splits processed_img folder contents into test, train, and val, saving these folders in binary_data2. In each folder there is other and food folders. Within other and foods folder there are folders with labels containing images.

binary_data2
├── train
│   ├── other
|	   ├── glass
|		  ├──glass1.img
│   └── food
└── test
    ├── other
    └── food
...


dataSplitReg.py: splits processed_img folder contents into test, train, and val, saving these folders in binary_data3. In each folder there is other and food folders. Within other and foods folder there are a mix of images from processed_img folder.

binary_data3
├── train
│   ├── other
|	   ├── glass1.img
	   ├── knife.img
│   └── food
└── test
    ├── other
    └── food
...


binaryModel.py: Binary model for is food or is not food. 

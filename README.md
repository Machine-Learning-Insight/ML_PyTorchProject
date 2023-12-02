# ML_PyTorchProject

How to Execute the Code:

Prerequisites
Python 3.x installed.
PyTorch and OpenCV libraries installed.
YOLOv8 model downloaded.
Access to the GitHub repository of the project. (Project code will be attached as zip file)
 
Installation Steps
Environment Setup
Ensure Python 3.x is installed on your system.
Install PyTorch: Visit the PyTorch official website and follow the installation instructions suitable for your system.
Install OpenCV: pip install opencv-python
install the YOLOv8:
Install the ultralytics package from PyPI: pip install ultralytics
Install YOLOv5 dependencies: check the official doc
Data Preparation
Download the annotated firearm dataset to a known directory.
Preprocess the dataset as necessary (resizing, normalization, etc.).
Model Training (If applicable)
Navigate to your training script directory.
Execute the training script: python NN_modelTester.py --data <dataset-path> --cfg <model-config> --weights <weights-path> --epochs <number-of-epochs>
Running the Weapon Detection System
Open the weapon detection script.
Configure the script to link to the video feed source and the trained model.
Run the script: python Detection.py
Real-Time Detection
The system will process the video feed in real-time.
Detected weapons will be identified with bounding boxes.
Troubleshooting
Confirm all dependencies are properly installed.
Verify the dataset paths in the scripts.
For YOLOv8-related issues, refer to the official YOLOv8 GitHub repository's issues section.

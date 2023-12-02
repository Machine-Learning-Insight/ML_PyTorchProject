# import onnx
from ultralytics import YOLO

import cv2
import math
# import torch
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

# import os
# os.kill(os.getpid(), 9)

def detect_objects(model, frame):
    
    # Perform inference on an image
    # Convert frame to YOLOv8 format
    results = model(frame)

    # Create a list to store objects
    objects = []

    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.numpy()
    # Extract classes, names, and confidences
    # boxes = results[0].boxes.xyxy.tolist()
    # classes = results[0].boxes.cls.tolist()
    # names = results[0].names
    # confidences = results[0].boxes.conf.tolist()

    for box in boxes:
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        obj = frame[y1:y2, x1:x2]
        objects.append(obj)

    return objects



def save_objects(objects, output_path, frame_number):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, obj in enumerate(objects):
        # Resize the object image to 32x32 pixels
        resized_obj = cv2.resize(obj, (32, 32))

        output_file = f"{output_path}/frame_{frame_number}_object_{i}.jpg"
        success = cv2.imwrite(output_file, resized_obj)
        if not success:
            print(f"Error saving {output_file}")


def process_video(video_path, output_path, model, frame_rate=5):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Process video
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process at specified frame rate
        if count % math.floor(fps / frame_rate) == 0:
            detected_objects = detect_objects(model, frame)
            save_objects(detected_objects, output_path, count)

        count += 1

    cap.release()
    cv2.destroyAllWindows()


# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
# results = model.val()

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
# success = model.export(format='onnx')

# Run interface on the source (detect objects from webcam video, I have tested; it's working!)
# results = model(source=0, show=True, conf=0.4,save=True)


# Load the pre-trained YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')


#Non-Gun video path
# video_path = 'Objects_Detection/sample-video.mp4'

#Gun video path
video_path = 'Objects_Detection/gun-video.mp4'

output_path = 'objects_frames'
process_video(video_path, output_path, model)
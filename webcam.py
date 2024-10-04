import cv2
import torch
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('best.pt')

# Open the webcam (change the argument if you're using a different camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    # If frame was not captured correctly, break the loop
    if not ret:
        break

    # Resize the frame for faster processing (optional, helps on low-powered devices)
    resized_frame = cv2.resize(frame, (640, 480))

    # Run YOLOv8 model on the frame
    results = model(resized_frame)  # Inference 

    # Annotate the frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the resulting frame with detections
    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()

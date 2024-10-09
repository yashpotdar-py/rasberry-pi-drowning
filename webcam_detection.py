import cv2
from ultralytics import YOLO

# Load the YOLOv8 Nano model for faster and lighter inference (if applicable)
# Use 'yolov8n.pt' for a smaller pre-trained model if you have one
model = YOLO('models\\drowning_detection_yolov8\\weights\\best.pt')

# Open the webcam (adjust the index if necessary)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_skip = 3  # Process every 3rd frame to reduce load
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame was not captured correctly, exit the loop
    if not ret:
        break

    # Process every 3rd frame to reduce CPU/GPU load
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Optionally resize the frame (only if needed, otherwise omit this)
    # resized_frame = cv2.resize(frame, (640, 480))  # Optional resizing if the image is large
    resized_frame = frame  # Skip resizing for performance

    # Run YOLOv8 model on the frame (Inference)
    # Avoid verbose output for lightweight operations
    results = model(resized_frame, verbose=False)

    # Draw the detections (can be omitted if just looking for faster inference)
    # Set line width to minimum for lighter processing
    annotated_frame = results[0].plot(line_width=1)

    # Display the resulting frame with detections
    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()

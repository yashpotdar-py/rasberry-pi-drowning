import cv2
import numpy as np

# Load the ONNX model using OpenCV
model_path = 'best.onnx'
net = cv2.dnn.readNetFromONNX(model_path)

# Open the webcam (adjust the index if necessary)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Resize input to smaller resolution to reduce resource usage
input_size = (640, 640)  # Lower resolution to reduce CPU load

# Define a function to preprocess the input frame


def preprocess_frame(frame, input_size=(640, 640)):
    # Resize frame to model input size and normalize pixel values
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1/255.0, size=input_size, swapRB=True, crop=False)
    return blob

# Define a function to post-process the output


def postprocess(frame, outputs, conf_threshold=0.5, nms_threshold=0.4):
    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    # Extract predictions
    for output in outputs[0]:
        scores = output[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            # Get the box coordinates, and scale to original frame size
            box = output[0:4] * np.array([w, h, w, h])
            (center_x, center_y, width, height) = box.astype("int")

            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Apply Non-Max Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)
    return [(boxes[i[0]], confidences[i[0]], class_ids[i[0]]) for i in indices]

# Define a function to draw bounding boxes and labels on the frame


def draw_detections(frame, detections, labels):
    for box, confidence, class_id in detections:
        x, y, w, h = box
        color = (0, 255, 0)  # Green for bounding box
        label = f"{labels[class_id]}: {confidence:.2f}"

        # Draw the rectangle and the label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Class labels for your custom model (adjust based on your classes)
labels = ['Active drowning', 'Possible Passive Drowner',
          'Swimming']  # Replace with actual class labels

# Reduce the frame rate by processing every nth frame
frame_skip = 4
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to reduce processing load
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Preprocess the frame for the ONNX model
    blob = preprocess_frame(frame, input_size=input_size)

    # Set the blob as input to the network
    net.setInput(blob)

    # Run forward pass to get the outputs
    outputs = net.forward()

    # Post-process the outputs to extract bounding boxes and labels
    detections = postprocess(frame, outputs)

    # Draw the detections on the frame
    draw_detections(frame, detections, labels)

    # Display the frame with detections
    cv2.imshow('YOLOv8 ONNX Webcam Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()

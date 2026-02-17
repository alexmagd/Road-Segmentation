import cv2
import numpy as np
import tensorflow as tf

# Load the trained U-Net model
model = tf.keras.models.load_model(r'Thesis\RoadSegmentation\roadSegmentationUnet.keras')

# Parameters
img_height = 512
img_width = 512

def preprocess_frame(frame):
    frame = cv2.resize(frame, (img_width, img_height))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def postprocess_mask(mask):
    mask = (mask > 0.5).astype(np.uint8)  # Apply threshold to get binary mask
    mask = mask[0]  # Remove the batch dimension
    mask_classes = []  # List to store masks for each class

    # Loop through each class
    for i in range(mask.shape[-1]):
        class_mask = mask[:, :, i]
        class_mask = cv2.resize(class_mask, (frame_width, frame_height))
        class_mask = np.expand_dims(class_mask, axis=-1)
        mask_classes.append(class_mask)

    return mask_classes

# Load the video
video_path = r'Thesis\RoadSegmentation\car driving.mp4'
cap = cv2.VideoCapture(video_path)

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define colors for each class
colors = [(255, 0, 0), (0, 255, 0)]  # Red and green colors for two classes
alpha = 0.5  # Transparency level (0: fully transparent, 1: fully opaque)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Predict the mask
    predicted_mask = model.predict(preprocessed_frame)

    # Postprocess the mask
    masks = postprocess_mask(predicted_mask)

    # Combine original frame and masks
    masked_frame = frame.copy()
    for i, mask in enumerate(masks):
        if i < len(colors):  # Ensure colors list has enough elements
            mask_colored = mask * colors[i]
            masked_frame = cv2.addWeighted(masked_frame, 1-alpha, mask_colored.astype(np.uint8), alpha, 0)

    # Write the frame to the output video
    out.write(masked_frame)

    # Display the masked frame
    cv2.imshow('Masked Frame', masked_frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, VideoWriter, and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()

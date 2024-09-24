import csv
import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Read the CSV file that contains image names and labels
input_csv = 'main_poses.csv'
data = pd.read_csv(input_csv)

# Function to extract keypoints from an image using MediaPipe
def extract_keypoints(image_path):
    # Read the image with OpenCV
    img = cv2.imread(image_path)

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to get pose landmarks
    result = pose.process(img_rgb)

    # Check if landmarks are detected
    if result.pose_landmarks:
        keypoints = []
        for landmark in result.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y])  # Only x and y coordinates
        return keypoints
    else:
        return [[0, 0]] * 33  # 33 keypoints in MediaPipe Pose (return [0,0] if no pose detected)

# Create CSV file
output_file = 'pose_keypoints_and_labels.csv'

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header for keypoints (e.g., keypoint_1_x, keypoint_1_y, ..., label)
    header = ['image_name'] + [f'keypoint_{i}_{coord}' for i in range(1, 34) for coord in ['x', 'y']] + ['label']
    writer.writerow(header)

    # Write data for each image
    for index, row in data.iterrows():
        image_path = f"./DataBase/poses/{row['image_name']}"
        keypoints = extract_keypoints(image_path)

        # Flatten keypoints into a list (alternating x and y)
        flattened_keypoints = [coord for keypoint in keypoints for coord in keypoint]
        writer.writerow([row['image_name']] + flattened_keypoints + [row['label']])

print(f"CSV file '{output_file}' created successfully!")

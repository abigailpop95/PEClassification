import csv
import os
import cv2
import mediapipe as mp
import pandas as pd


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)


input_csv = 'images.csv'
data = pd.read_csv(input_csv)

# extract keypoints from an image using MediaPipe
def extract_keypoints(image_path):

    img = cv2.imread(image_path)


    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = pose.process(img_rgb)


    if result.pose_landmarks:
        keypoints = []
        for landmark in result.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y])  #
        return keypoints
    else:
        return [[0, 0]] * 33 

# create CSV file
output_file = 'pose_keypoints_and_labels.csv'

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # write the header for keypoints 
    header = ['image_name'] + [f'keypoint_{i}_{coord}' for i in range(1, 34) for coord in ['x', 'y']] + ['label']
    writer.writerow(header)

    for index, row in data.iterrows():
        image_path = f"./images/{row['image_name']}"
        keypoints = extract_keypoints(image_path)

        flattened_keypoints = [coord for keypoint in keypoints for coord in keypoint]
        writer.writerow([row['image_name']] + flattened_keypoints + [row['label']])

print(f"CSV file '{output_file}' created successfully!")

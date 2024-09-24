import cv2
import mediapipe as mp
import os
import pandas as pd

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to extract pose landmarks and save the image with keypoints
def extract_pose_landmarks(image_path, output_image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None  # If image couldn't be loaded, return None

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Convert the BGR image to RGB before processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find pose landmarks
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save the image with keypoints to the output folder
            cv2.imwrite(output_image_path, image)

            # Extract the x, y, z coordinates for all landmarks
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            return keypoints  # Return list of (x, y, z) keypoints
        else:
            return None  # Return None if no pose landmarks were detected

# Function to process all images in a folder and save keypoints and images with keypoints
def process_images_in_folder(input_folder, output_csv='pose_data.csv', output_image_folder='output_images'):
    data = []
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

    # Create output folder if it doesn't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_image_folder, image_file)  # Path to save the image with keypoints

        # Extract pose landmarks and save the image with keypoints
        keypoints = extract_pose_landmarks(image_path, output_image_path)
        
        if keypoints:
            # Flatten the keypoints list (from [[x1, y1, z1], ...] to [x1, y1, z1, ...])
            keypoints_flat = [coord for point in keypoints for coord in point]
            data.append([image_file] + keypoints_flat)
        else:
            print(f"Pose not detected in {image_file}")

    # Save the data to a CSV
    # Update the column names to have 99 coordinates + 1 image column
    column_names = ['image'] + [f'{axis}{i}' for i in range(33) for axis in ['x', 'y', 'z']]

    # Now, when creating the DataFrame, it will match the expected 100 columns
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(output_csv, index=False)

    print(f"Pose data saved to {output_csv}")
    print(f"Images with keypoints saved to {output_image_folder}")

# Example usage
process_images_in_folder('DataBase/poses', output_csv='pose_data.csv', output_image_folder='DataBase/output_poses')  # Replace with the path to your folder

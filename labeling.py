import pandas as pd

# Load CSV
df = pd.read_csv('pose_data.csv')

# Function to label data (lying down = injured, based on multiple body keypoints)
def label_data(row):
    # Extract y-coordinates for key body landmarks
    # Left and right shoulders (landmarks 11, 12)
    left_shoulder_y = row['y11']
    right_shoulder_y = row['y12']

    # Left and right hips (landmarks 23, 24)
    left_hip_y = row['y23']
    right_hip_y = row['y24']

    # Left and right knees (landmarks 25, 26)
    left_knee_y = row['y25']
    right_knee_y = row['y26']

    # Left and right ankles (landmarks 27, 28)
    left_ankle_y = row['y27']
    right_ankle_y = row['y28']

    # Calculate average y-coordinate for the shoulders, hips, knees, and ankles
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    avg_hip_y = (left_hip_y + right_hip_y) / 2
    avg_knee_y = (left_knee_y + right_knee_y) / 2
    avg_ankle_y = (left_ankle_y + right_ankle_y) / 2

    # Check if the y-coordinates are aligned, indicating a lying down posture
    threshold = 0.1  # Adjust this threshold as needed based on the scale of y-coordinates

    # Check for 'Injured' (lying down)
    if (
        abs(avg_shoulder_y - avg_hip_y) < threshold and
        abs(avg_hip_y - avg_knee_y) < threshold and
        abs(avg_knee_y - avg_ankle_y) < threshold
    ):
        return 'Injured'

    # Check for 'Not Injured' (standing or sitting, based on large differences in y-coordinates)
    elif (
        abs(avg_shoulder_y - avg_hip_y) > threshold * 1.5 or
        abs(avg_hip_y - avg_knee_y) > threshold * 1.5 or
        abs(avg_knee_y - avg_ankle_y) > threshold * 1.5
    ):
        return 'Not Injured'

    # If the posture doesn't fit either case, label it as 'Unknown'
    else:
        return 'Unknown'

# Apply labeling function to each row
df['label'] = df.apply(label_data, axis=1)

# Save the labeled dataset for further training
df.to_csv('labeled_pose_data.csv', index=False)

print("Labeled dataset saved to 'labeled_pose_data.csv'.")

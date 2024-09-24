import pandas as pd

# Load CSV
df = pd.read_csv('pose_data.csv')

# Function to dynamically check which points exist in the dataset
def label_data(row):
    # Find which 'x' coordinates are present in the dataset
    detected_points = [col for col in row.index if col.startswith('x')]
    num_detected_points = sum(~pd.isna(row[detected_points]))  # Count how many landmarks have non-NaN values

    # First check: label as "unknown" if fewer than 25 points are detected
    if num_detected_points < 25:
        return 'Unknown'

    # Check if shoulder and hip z-coordinates exist (adjust based on your dataset structure)
    if {'z11', 'z12', 'z23', 'z24'}.issubset(row.index):
        # Extract z-coordinates for the hips and shoulders
        left_shoulder_z = row['z11']
        right_shoulder_z = row['z12']
        left_hip_z = row['z23']
        right_hip_z = row['z24']

        # Calculate average z-coordinates for shoulders and hips
        avg_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2
        avg_hip_z = (left_hip_z + right_hip_z) / 2

        # Define a threshold for checking if the person is lying down
        z_threshold = 0.1  # Adjust this threshold based on the scale of z-values in your dataset

        # Second check: label as "badly injured" if z-coordinates of shoulders and hips are close
        if abs(avg_shoulder_z - avg_hip_z) < z_threshold:
            return 'Badly Injured'
        else:
            return 'Not Injured or Slightly Injured'
    
    # Default label if none of the conditions are met
    return 'Unknown'

# Apply labeling function to each row
df['label'] = df.apply(label_data, axis=1)

# Save the labeled dataset for further analysis
df.to_csv('labeled_pose_data.csv', index=False)

print("Labeled dataset saved to 'labeled_pose_data.csv'.")

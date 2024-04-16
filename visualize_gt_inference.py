import cv2
import os
import argparse

# Function to read values from file
def read_values_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        values = [(line.strip()) for line in lines]
    return values

# Function to display video with shapes
def display_video_with_shapes(frames_folder, values_file, output_file):
    # Read all image file names from the folder
    frame_files = sorted(os.listdir(frames_folder))[18:]

    # Read values from file
    values = read_values_from_file(values_file)

    # Initialize video writer
    frame_height, frame_width, _ = cv2.imread(os.path.join(frames_folder, frame_files[0])).shape
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    for frame_file, value in zip(frame_files, values):
        # Read frame
        frame = cv2.imread(os.path.join(frames_folder, frame_file))

        # Draw shape based on value
        # if value == 1:  # Example condition, you can replace it with your own logic
        #     cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)  # Example rectangle
        cv2.putText(frame, str(value), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Add more conditions for other shapes as needed

        # Write frame to output video
        out.write(frame)

    # Release video writer
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="100", help="Path to the folder containing frames")
    parser.add_argument("--actions", type=str, default="actions.txt", help="Path to the txt file containing actions")
    parser.add_argument("--output_dir", type=str, default="gt_inference", help="Output video file")
    args = parser.parse_args()

    frames_folder = f'/home/punygod_admin/SoundSense/soundsense/data/mulsa/data/{args.run_id}/video/'
    values_file = f'/home/punygod_admin/SoundSense/soundsense/gt_inference/{args.run_id}/actions.txt'
    output_file = f'/home/punygod_admin/SoundSense/soundsense/{args.output_dir}/{args.run_id}/inference.mp4'
    display_video_with_shapes(frames_folder, values_file, output_file)

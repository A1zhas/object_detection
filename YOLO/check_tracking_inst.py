import cv2
from ultralytics import YOLO
import random
import numpy as np

def process_video_with_tracking_segment(model, input_video_path, show_video=True, save_video=False, output_video_path="output_video.avi"):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file at {input_video_path}.")
    else:
        print("Video file opened successfully.")
    
    # Get input video frame rate and dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Try 'DIVX' for .avi output
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read a frame from the video.")
            break
        else:
            print(f"Frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} read successfully.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform object tracking
        results = model.track(frame_rgb, iou=0.5, conf=0.70, persist=True, imgsz=640, verbose=False, tracker="botsort.yaml")
        
        # Check if boxes exist and simplify processing
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):
                # Generate a random color for each object based on its ID
                random.seed(int(id))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # Draw the bounding box and ID
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f"ID {id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if save_video:
            out.write(frame)

        if show_video:
            frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            cv2.imshow("frame", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video capture and writer
    cap.release()
    if save_video:
        out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Example usage:
model = YOLO('C:\\Users\\Alen\\Desktop\\CV\\Object_detection\\YOLO\\runs\\segment\\train5\\weights\\best.pt')
process_video_with_tracking_segment(model, "C:\\Users\\Alen\\Desktop\\CV\\Object_detection\\YOLO\\track_video.mp4", 
                                    show_video=True, save_video=True, output_video_path="C:\\Users\\Alen\\Desktop\\CV\\Object_detection\\YOLO\\output_video_id.avi")

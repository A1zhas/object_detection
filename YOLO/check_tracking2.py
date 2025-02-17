import cv2
from ultralytics import YOLO
import random
import numpy as np
import sys, os


def process_video_with_tracking(model, input_video_path, show_video=True, save_video=False, output_video_path="output_video.mp4"):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get input video frame rate and dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps/2, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, iou=0.6, conf=0.75, persist=True, imgsz=640, verbose=False, tracker="botsort.yaml")

        if results[0].boxes.id != None: # this will ensure that id is not None -> exist tracks
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):
                # Generate a random color for each object based on its ID
                random.seed(int(id))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 4)
                cv2.putText(
                    frame,
                    f"Id {id}",
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    2,
                )

        if save_video:
            out.write(frame)

        if show_video:
            frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            cv2.imshow("frame", frame)

        if cv2.waitKey(int(1000/fps)) & 0xFF == ord("q"):
            break

    # Release the input video capture and output video writer
    cap.release()
    if save_video:
        out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Example usage:
model = YOLO('C:\\Users\\Alen\\Desktop\\CV\\Object detection\\YOLO\\runs\\segment\\train5\\weights\\best.pt')
process_video_with_tracking(model, input_video_path="C:\\Users\\Alen\\Desktop\\CV\\Object detection\\YOLO\\track_video.mp4",
                                          show_video=True, save_video=False,
                                          output_video_path="C:\\Users\\Alen\\Desktop\\CV\\Object detection\\YOLO\\output_video.mp4")

from ultralytics import YOLO
import cv2
from patched_yolo_infer import visualize_results_usual_yolo_inference

# Initialize a YOLO-World model
model = YOLO('yolov8l-world.pt')

# Define custom classes
model.set_classes(["person", "table", "tea cup"])

# Load the image
img_path = 'C:\\Users\\Alen\\Desktop\\CV\\Object_detection\\ZERO_SHOT\\images\\1.jpg'
img = cv2.imread(img_path)

# Check if the image is loaded successfully
if img is None:
    print("Error: Image not loaded correctly. Check the path.")
else:
    # Visualize the results using the provided custom function
    output_img = visualize_results_usual_yolo_inference(
        img,
        model,
        imgsz=640,
        conf=0.2,
        iou=0.7,
        segment=False,
        delta_colors=0,
        thickness=4,
        font_scale=1.0,
        show_boxes=True,
        random_object_colors=False,
        show_confidences=False,
        show_class=True
    )

    # Check if the output image is valid
    if output_img is not None:
        # Display the resulting image (you can manually save it after)
        cv2.imshow("Detected Image", output_img)
        cv2.waitKey(0)  # Wait for a key press to close the image window
        cv2.destroyAllWindows()
    else:
        print("Error: No output image generated.")

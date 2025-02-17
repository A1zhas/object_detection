import cv2
from ultralytics import YOLO

def process_and_save_video(model, input_video_path, show_video=True, save_video=False, output_video_path="output_video.mp4"):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Определяем видеокодек и создаем объект VideoWriter
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Попробуй 'XVID', если не работает
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Запускаем модель YOLOv8 с трекером
        results = model.track(frame, persist=True, iou=0.65, conf=0.70,
                              tracker="botsort.yaml", imgsz=640, verbose=False)

        annotated_frame = results[0].plot()

        # Сохраняем кадр в видеофайл
        if save_video:
            out.write(annotated_frame)

        # Показываем результат
        if show_video:
            small_frame = cv2.resize(annotated_frame, (frame_width // 2, frame_height // 2))
            cv2.imshow("YOLOv8 Tracking", small_frame)

        # Выход по нажатию "q"
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    if save_video and out is not None:
        out.release()
    cv2.destroyAllWindows()

# Использование
model = YOLO('C:\\Users\\Alen\\Desktop\\CV\\Object detection\\YOLO\\runs\segment\\train5\\weights\\best.pt')
process_and_save_video(model, input_video_path="C:\\Users\\Alen\\Desktop\\CV\\Object detection\\YOLO\\track_video.mp4",
                       show_video=True, save_video=False,
                       output_video_path="C:\\Users\\Alen\\Desktop\\CV\\Object detection\\YOLO\\output_video.mp4")

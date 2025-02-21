import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Загрузка модели YOLO
model = YOLO(r'C:\Users\Alen\Desktop\CV\Object_detection\YOLO\runs\segment\train5\weights\best.pt')

# Загрузка изображения
image = cv2.imread('C:\Users\Alen\Desktop\CV\Object_detection\YOLO\photo\IMG_2957.JPG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование в RGB для matplotlib

# Параметры инференса
alpha = 0.2
iou = 0.65
conf = 0.15
imgsz = 608

# Выполнение инференса
results = model(image, imgsz=imgsz, iou=iou, conf=conf, verbose=False)

# Проверяем, есть ли обнаруженные маски
if results[0].masks is None:
    print("Нет обнаруженных масок.")
    plt.imshow(image)
    plt.axis('off')
    plt.show()
else:
    # Получение бинарных масок и их количество
    masks = results[0].masks.data.cpu().numpy()  # Переводим в numpy
    num_masks = masks.shape[0]

    # Определение случайных цветов
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]

    # Создание слоя масок
    mask_overlay = np.zeros_like(image, dtype=np.uint8)

    # Наложение масок
    for i in range(num_masks):
        color = colors[i]  # Случайный цвет
        mask_resized = cv2.resize(masks[i], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Добавляем цветной слой к маске
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[mask_resized > 0] = color
        mask_overlay = cv2.addWeighted(mask_overlay, 1, color_mask, alpha, 0)

    # Объединение исходного изображения и масок
    result_image = cv2.addWeighted(image, 1, mask_overlay, 0.6, 0)

    # Отображение результата
    plt.figure(figsize=(8, 8), dpi=150)
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()

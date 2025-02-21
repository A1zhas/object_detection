from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = YOLO('C:\\Users\\Alen\\Desktop\\CV\\Object_detection\\YOLO\\runs\\segment\\train5\\weights\\best.pt')

img = cv2.imread('C:\\Users\\Alen\\Desktop\\CV\\Object_detection\\YOLO\\photo\\IMG_2957.JPG')

results = model(img, imgsz=640, iou=0.4, conf=0.7, verbose=True)

# Извлечение координат боксов
boxes = results[0].boxes.xywhn

print(boxes)

# Проверка формы (количество объектов)
print(boxes.shape)

# Имена классов для каждого объекта
print(results[0].names)

# Индексы классов для каждого объекта
print(results[0].boxes.cls)

# Печать уверенности для каждого бокса
print(results[0].boxes.conf)

# Отображение маски сегментации для объекта с индексом 5
plt.imshow(results[0].masks.data[5].cpu().numpy(), 'gray')
plt.show()



# Получение классов и имен классов
classes = results[0].boxes.cls.cpu().numpy()
class_names = results[0].names

# Получение бинарных масок и их количество
masks = results[0].masks.data  # Формат: [число масок, высота, ширина]
num_masks = masks.shape[0]

# Определение случайных цветов и прозрачности для каждой маски
colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]  # Случайные цвета

# Создание изображения для отображения масок
mask_overlay = np.zeros_like(img)

labeled_image = img.copy()

# Добавление подписей к маскам
for i in range(num_masks):
    color = colors[i]  # Случайный цвет
    mask = masks[i].cpu()

    # Изменение размера маски до размеров исходного изображения с использованием метода ближайших соседей
    mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Получение класса для текущей маски
    class_index = int(classes[i])
    class_name = class_names[class_index]

    # Добавление подписи к маске
    mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Отрисовка контуров маски
    for contour in mask_contours:
        cv2.drawContours(labeled_image, [contour], -1, color, 5)

    # Добавление имени класса (подписи) на маску
    if len(mask_contours) > 0:
        contour_center = np.mean(mask_contours[0], axis=0).astype(int)
        cv2.putText(labeled_image, class_name, (contour_center[0][0], contour_center[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Отобразите итоговое изображение с наложенными масками и подписями
plt.figure(figsize=(8, 8), dpi=150)
labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
plt.imshow(labeled_image)
plt.axis('off')
plt.show()
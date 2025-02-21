from PIL import Image
from samgeo.text_sam import LangSAM

# Инициализация модели
model = LangSAM()

# Текстовый запрос
text_prompt = 'lamp'

# Путь к изображению
image_path = "C:/Users/Alen/Desktop/CV/Object_detection/ZERO_SHOT/images/1.jpg"

# Загружаем изображение
image_pil = Image.open(image_path)

# Выполняем предсказание с использованием модели
model.predict(image_path, text_prompt, box_threshold=0.25, text_threshold=0.3)

# Параметры отображения сегментации
model.show_anns(
    cmap='Reds',
    box_color='red',
    title=f'Автоматическая сегментация {text_prompt}',
    alpha=0.5,
    blend=True,
    output="C:/Users/Alen/Desktop/CV/Object_detection/ZERO_SHOT/images/result.jpg"  # Путь для сохранения
)

print("Сегментация завершена и результат сохранен.")

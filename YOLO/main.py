from ultralytics import YOLO
import torch
import yaml  # Добавим импорт библиотеки yaml


def main():
    yaml_file_path = "dataset/data.yaml"
    with open(yaml_file_path, 'r') as file:
        dataset_info = yaml.safe_load(file)

    print("Путь к обучающим данным:", dataset_info['train'])
    print("Путь к валидационным данным:", dataset_info['val'])
    print("Количество классов:", dataset_info['nc'])
    print("Имена классов:", dataset_info['names'])

    # Загружаем модель .pt вместо .yaml
    model = YOLO('yolo11s-seg.pt')

    # Обучение модели
    results = model.train(
        data='dataset/data.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # Сохранение обученной модели
    model.export(format='onnx')


if __name__ == '__main__':
    main()

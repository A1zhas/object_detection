import torch

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
else:
    print("No CUDA-compatible GPU found!")



device = "cuda" if torch.cuda.is_available() else "cpu"

# Создаём случайный тензор и перемещаем на GPU
x = torch.rand((1000, 1000), device=device)
y = torch.rand((1000, 1000), device=device)

# Выполняем умножение матриц
result = torch.matmul(x, y)

print(f"Результат вычислений на {device}: {result.shape}")

# Проверяем потребление памяти после загрузки данных
print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

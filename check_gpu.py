import torch
cuda = torch.cuda.is_available()
print(f'CUDA доступна: {cuda}')
if cuda:
    print(f'Устройство: {torch.cuda.get_device_name(0)}')
    print(f'Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
else:
    print('Используется CPU')
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Используется GPU")
else:
    device = torch.device('cpu')
    print("Используется CPU")

import torch

print("Используется GPU:", torch.cuda.is_available())
print("Количество доступных GPU:", torch.cuda.device_count())

print("Используется GPU:", torch.cuda.is_available())
print("Количество доступных GPU:", torch.cuda.device_count())
print("Имя GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Нет доступного GPU")

import platform
print(platform.architecture())
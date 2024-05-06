import torch
import torch.nn as nn
import torch.optim as optim

# Определение нейронной сети
class LandscapeClassifier(nn.Module):
    def __init__(self):
        super(LandscapeClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Входной слой размером 3 и скрытый слой размером 64
        self.fc2 = nn.Linear(64, 32)  # Скрытый слой размером 64 и 32
        self.fc3 = nn.Linear(32, 3)  # Выходной слой размером 32 и 3 (3 класса: гора, поле, впадина)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Применение функции активации ReLU к выходу первого слоя
        x = torch.relu(self.fc2(x))  # Применение функции активации ReLU ко второму слою
        x = self.fc3(x)  # Выходной слой без функции активации, так как используем CrossEntropyLoss
        return x

# Создание экземпляра нейронной сети
model = LandscapeClassifier()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()  # Функция потерь для многоклассовой классификации
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Стохастический градиентный спуск

# Генерация случайных данных
# Предположим, у нас есть 100 ландшафтных элементов
num_samples = 100

# Генерация высоты (от 0 до 500 м)
heights = torch.randint(0, 501, size=(num_samples, 1), dtype=torch.float32)

# Генерация площади (от 0 до 10000 кв. м)
areas = torch.randint(0, 10001, size=(num_samples, 1), dtype=torch.float32)

# Генерация среднего количества дождливых дней (от 50 до 300 дней)
rainy_days = torch.randint(50, 301, size=(num_samples, 1), dtype=torch.float32)

# Соединение всех характеристик в одну матрицу
features = torch.cat((heights, areas, rainy_days), dim=1)

# Генерация соответствующих классов (0 - гора, 1 - поле, 2 - впадина)
# Пусть горы имеют среднюю высоту выше 300 м, поля - площадь больше 5000 кв. м, а впадины - количество дождливых дней меньше 150
classes = torch.zeros(num_samples, dtype=torch.long)  # начинаем с класса 0 (гора)
classes[(heights.squeeze() <= 300) & (areas.squeeze() > 5000)] = 1  # класс 1 (поле)
classes[rainy_days.squeeze() < 150] = 2  # класс 2 (впадина)

# Вывод данных
print("Height (m):\n", heights[:5])
print("\nArea (sq. m):\n", areas[:5])
print("\nNumber of rainy days (days):\n", rainy_days[:5])
print("\nClasses:\n", classes[:5])


# Обучение нейронной сети
# Здесь будет ваш код обучения с использованием данных


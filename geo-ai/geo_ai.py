import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ����������� ��������� ����
class LandscapeClassifier(nn.Module):
    def __init__(self):
        super(LandscapeClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # ������� ���� �������� 3 � ������� ���� �������� 64
        self.fc2 = nn.Linear(64, 32)  # ������� ���� �������� 64 � 32
        self.fc3 = nn.Linear(32, 3)  # �������� ���� �������� 32 � 3 (3 ������: ����, ����, �������)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ���������� ������� ��������� ReLU � ������ ������� ����
        x = torch.relu(self.fc2(x))  # ���������� ������� ��������� ReLU �� ������� ����
        x = self.fc3(x)  # �������� ���� ��� ������� ���������, ��� ��� ���������� CrossEntropyLoss
        return x

# �������� ���������� ��������� ����
model = LandscapeClassifier()

# ����������� ������� ������ � ������������
criterion = nn.CrossEntropyLoss()  # ������� ������ ��� �������������� �������������
optimizer = optim.SGD(model.parameters(), lr=0.01)  # �������������� ����������� �����

# ��������� ��������� ������
# �����������, � ��� ���� 100 ����������� ���������
num_samples = 100

# ��������� ������ (�� 0 �� 500 �)
heights = torch.randint(0, 501, size=(num_samples, 1), dtype=torch.float32)

# ��������� ������� (�� 0 �� 10000 ��. �)
areas = torch.randint(0, 10001, size=(num_samples, 1), dtype=torch.float32)

# ��������� �������� ���������� ��������� ���� (�� 50 �� 300 ����)
rainy_days = torch.randint(50, 301, size=(num_samples, 1), dtype=torch.float32)

# ���������� ���� ������������� � ���� �������
features = torch.cat((heights, areas, rainy_days), dim=1)

# ��������� ��������������� ������� (0 - ����, 1 - ����, 2 - �������)
# ����� ���� ����� ������� ������ ���� 300 �, ���� - ������� ������ 5000 ��. �, � ������� - ���������� ��������� ���� ������ 150
classes = torch.zeros(num_samples, dtype=torch.long)  # �������� � ������ 0 (����)
classes[(heights.squeeze() <= 300) & (areas.squeeze() > 5000)] = 1  # ����� 1 (����)
classes[rainy_days.squeeze() < 150] = 2  # ����� 2 (�������)

# ����� ������
print("Height (m):\n", heights[:5])
print("\nArea (sq. m):\n", areas[:5])
print("\nNumber of rainy days (days):\n", rainy_days[:5])
print("\nClasses:\n", classes[:5])

# �������� ��������� ����
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(features)
    loss = criterion(outputs, classes)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting the training loss
plt.plot(losses, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

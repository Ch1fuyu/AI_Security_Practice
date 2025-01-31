import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import os

# 定义LeNet模型（与攻击代码一致）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 加载MNIST数据集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=64,
    shuffle=True
)

# 初始化模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# 训练模型
epochs = 20
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch+1}/{epochs} | Batch: {batch_idx} | Loss: {loss.item():.4f}")

# 保存模型权重
# 确保目标文件夹存在
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# 保存模型到指定文件夹
model_path = os.path.join(output_dir, "lenet_mnist_model.pth")
torch.save(model.state_dict(), model_path)
print(f"模型已保存为 {model_path}")
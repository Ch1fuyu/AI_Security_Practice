import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# 定义LeNet模型
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

# 加载MNIST测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',
                   train=False,
                   download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=1,
    shuffle=True
)

# 加载预训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
pretrained_model = "../models/lenet_mnist_model.pth"  # 注意路径需与本地一致
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()

# 定义FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 约束像素值在[0,1]
    return perturbed_image

# 测试攻击效果
def test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue  # 跳过初始预测错误的样本
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output_after = model(perturbed_data)
        final_pred = output_after.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
        # 记录对抗样本
        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy: {correct}/{len(test_loader)} = {final_acc}")
    return final_acc, adv_examples

# 执行攻击实验
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
accuracies = []
examples = []
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 1. 加载预训练的 InceptionV3 模型
model = InceptionV3(weights='imagenet')  # 使用 ImageNet 数据集预训练的权重

# 2. 加载图片并进行预处理
img_path = "C:\\Users\Chifuyu\Desktop\D\学习\研\汇报\\2025.3\data\poison_s4.0.png"  # 图片路径
img = image.load_img(img_path, target_size=(299, 299))  # InceptionV3 输入图片大小为 299x299
img_array = image.img_to_array(img)  # 将图片转换为 NumPy 数组
img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
img_array = preprocess_input(img_array)  # 对图片进行预处理

# 3. 使用模型进行预测
predictions = model.predict(img_array)

# 4. 解码预测结果
decoded_predictions = decode_predictions(predictions, top=10)[0]  # 获取前 5 个最可能的类别
print("Prediction Results:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:0.2f})")

# 5. 提取类别名称和概率
labels = [pred[1] for pred in decoded_predictions]  # 提取类别名称
probabilities = [pred[2] for pred in decoded_predictions]  # 提取概率

# 6. 创建一个大图，包含测试图片和概率分布表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))  # 创建两个子图

# 6.1 显示测试图片
ax1.imshow(img)
ax1.set_title(f'Top Prediction: {decoded_predictions[0][1]}\nImage Size: {img.size[0]}x{img.size[1]}', fontsize=12)
ax1.set_xlabel(f'Width: {img.size[0]} pixels', fontsize=10)
ax1.set_ylabel(f'Height: {img.size[1]} pixels', fontsize=10)
ax1.set_xticks(range(0, img.size[0], 50))  # 设置X轴刻度为50像素
ax1.set_yticks(range(0, img.size[1], 50))  # 设置Y轴刻度为50像素
ax1.grid(True, linestyle='--', alpha=0.5)  # 添加网格线

# 6.2 绘制概率分布表
ax2.bar(labels, probabilities, color='skyblue')
ax2.set_xlabel('Class Labels', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Prediction Probability Distribution', fontsize=14)
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

# 7. 调整布局并显示
plt.tight_layout()
plt.show()
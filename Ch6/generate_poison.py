import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 加载预训练的 InceptionV3 模型
model = InceptionV3(weights='imagenet', include_top=True)

# 加载两张图片并进行预处理
img_path = "C:\\Users\Chifuyu\Desktop\D\学习\研\汇报\\2025.3\data\cat_origin.png"  # 替换为您的第一张图片路径
img2_path = "C:\\Users\Chifuyu\Desktop\D\学习\研\汇报\\2025.3\data\washbasin.png"  # 替换为您的第二张图片路径

img = image.load_img(img_path, target_size=(299, 299))
img2 = image.load_img(img2_path, target_size=(299, 299))

img_array = image.img_to_array(img)
img2_array = image.img_to_array(img2)

img_array = np.expand_dims(img_array, axis=0)
img2_array = np.expand_dims(img2_array, axis=0)

img_array = img_array.astype('float32') / 255.0  # 归一化到 [0, 1]
img2_array = img2_array.astype('float32') / 255.0  # 归一化到 [0, 1]

# 获取第二张图片 img2 对应的类别标签
img2_logits = model.predict(img2_array)
img2_pred = np.argmax(img2_logits, axis=1)
img2_class = tf.keras.utils.to_categorical(img2_pred, num_classes=1000)

# 定义模型训练参数值
demo_epsilon = 4.0 / 255.0 # def 2
demo_lr = 0.1 # def 0.1
demo_steps = 100 # def 100

# 将图像转换为 TensorFlow 变量
x_hat = tf.Variable(img_array, dtype=tf.float32)

# 定义损失函数
def loss_fn(x_hat, y_tar):
    x_initial = tf.constant(img_array)
    loss_ssim = (1 - tf.reduce_mean(tf.image.ssim(x_hat, x_initial, max_val=1.0))) / 2.0
    loss_ssim = tf.square(loss_ssim)

    logits = model(x_hat, training=False)
    loss_soft = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_tar)
    return loss_ssim + loss_soft

# 优化器
optimizer = tf.optimizers.SGD(learning_rate=demo_lr)

# 投影步骤
def project_step(x_hat, epsilon):
    below = x_hat - epsilon
    above = x_hat + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    return projected

# 执行训练和投影步骤
for i in range(demo_steps):
    with tf.GradientTape() as tape:
        loss = loss_fn(x_hat, img2_class)
    gradients = tape.gradient(loss, x_hat)
    optimizer.apply_gradients([(gradients, x_hat)])
    x_hat.assign(project_step(x_hat, demo_epsilon))
    if (i + 1) % 10 == 0:
        print(f'step {i + 1}, loss={loss.numpy()}')

# 获取加入了扰动的图像
perturbed_img = x_hat.numpy()
perturbed_img = np.squeeze(perturbed_img, axis=0)

# 显示原始图像和加入了扰动的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(img_array, axis=0))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(perturbed_img)
plt.title('Perturbed Image')
plt.axis('off')

plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 解决 matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 负号正常显示

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# 输入范围
x = np.linspace(-10, 10, 1000)

# 计算输出
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(x, y_sigmoid, label='Sigmoid 函数')
plt.plot(x, y_tanh, label='Tanh 函数')
plt.plot(x, y_relu, label='ReLU 函数')

# 图形设置
plt.xlabel('输入 x')
plt.ylabel('输出 f(x)')
plt.title('常用激活函数曲线对比')
plt.legend()

# 关闭网格线（不调用 plt.grid 或显式关闭）
plt.grid(False)

plt.show()

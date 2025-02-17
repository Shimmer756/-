import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 参数设置
hbar = 1.0  # 简化单位制，设定 hbar = 1
m = 1.0  # 粒子质量
omega = 1.0  # 谐振子频率
L = 5.0  # 定义计算区域的大小
N = 1000  # 离散化的格点数
dx = 2*L / N  # 每个格点的间距
x = np.linspace(-L, L, N)  # 离散的空间坐标

# 创建哈密顿量矩阵
T = -0.5 * hbar**2 / m * (np.diag(np.ones(N-1), 1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), -1)) / dx**2  # 动能部分
print(T)
print(np.diag(np.ones(N-1), -1))
print(np.diag(np.ones(N-1), 1))

V = 0.5 * m * omega**2 * np.diag(x**2)  # 势能部分
H = T + V  # 总哈密顿量

# 求解本征值问题
eigenvalues, eigenvectors = eigh(H)

# 绘制前几个本征波函数
num_wavefunctions = 5
plt.figure(figsize=(10, 6))
for n in range(num_wavefunctions):
    plt.plot(x, eigenvectors[:, n]**2, label=f"n = {n}, E = {eigenvalues[n]:.2f}")

plt.xlabel('Position (x)')
plt.ylabel('Probability Density')
plt.title('Wavefunctions of the 1D Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()

# 打印前几个能量本征值
print("Energy Eigenvalues (first few):")
for n in range(num_wavefunctions):
    print(f"n = {n}, E = {eigenvalues[n]:.3f}")


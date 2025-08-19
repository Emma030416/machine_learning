import torch
from torch.utils.tensorboard import SummaryWriter # 用于绘制loss曲线

# 能用cuda就放GPU计算，否则放CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成数据
inputs = torch.rand(100, 3) # 特征：随机生成一个shape为(100, 3)的tensor，共100条数据，每条3个feature，数值在0-1之间
weights = torch.tensor([[1.1],
                       [2.2],
                       [3.3]]) # 预设的权重
bias = torch.tensor(4.4) # 预设的偏置
targets = inputs @ weights + bias + 0.1 * torch.randn(100, 1) # 标签：先做矩阵乘法，(100,3)*(3,1)得(100,1)，再给每个加上偏置，最后加上噪声模拟真实场景

# 创建一个SummaryWriter实例
writer = SummaryWriter()

#      初始化参数            启动梯度追踪       放到相同设备上
w = torch.rand((3, 1), requires_grad=True, device=device) # 随机初始化权重矩阵
b = torch.rand((1,),   requires_grad=True, device=device) # 随机初始化偏置标量

# 将数据（特征和标签）迁移至对应设备
inputs = inputs.to(device)
targets = targets.to(device)

# 设置超参数
epoch = 10000 # 迭代次数
lr = 0.003 # 学习率

# 训练循环
for i in range(epoch):
    outputs = inputs @ w + b # 线性回归模型预测值
    loss = torch.mean(torch.square(outputs - targets)) # MSE均方误差
    print(f"loss:{loss.item()}")
    writer.add_scalar("loss/train", loss.item(), i)
    loss.backward() # 反向传播，计算梯度

    # 迭代更新参数
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

print(f"训练后的权重w:{w}")
print(f"训练后的偏置b:{b}")
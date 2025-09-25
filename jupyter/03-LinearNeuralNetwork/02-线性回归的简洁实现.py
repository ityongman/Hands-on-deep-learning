import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l


'''
业务处理顺序
1. 定义我们线性方程的系数、 参数, 并根据系数/参数 生成1000组实验数据 -- 均值=0, 方差=1 带噪声
2. 定义数据加载器, 将我们生成的数据保证成Pytorch的数据集, 从生成的数据中随机/顺序加载数据 (shuffle=True/False)
3. 定义我们的神经网络模型 这里是线性函数
4. 定义损失函数
5. 定义优化算法 SGD 随机梯度下降
6. 迭代样本数据, 来修正 w、b
'''

true_w = torch.tensor([2, -3.5])
true_b = 3.5
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_Array, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_Array)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 定义我们的神经网络模型 -- 线性模型
# Sequential 表示网络模型的层数, 这里是1层
net = nn.Sequential(nn.Linear(in_features=2, out_features=1))
nn.init.normal_(net[0].weight, mean=0, std=0.01)
nn.init.zeros_(net[0].bias) # 将偏置初始化为零

# loss = nn.MSELoss(reduction='mean')
loss = nn.MSELoss(reduction='sum')

# lr = 0.03 # learn rate 学习率
lr = 0.003 # learn rate 学习率 批次 batch_size=10 MSELoss 函数, mean=sum/batch_size, 如果reduction=sum， lr需要除以batch_size
trainer = torch.optim.SGD(net.parameters(), lr=lr)

num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X), y) # 计算损失
        trainer.zero_grad() # 梯度清零
        l.backward() # 计算梯度
        trainer.step() # 更新参数

    l = loss(net(features), labels) # 使用总样本计算损失
    print(f'epoch {epoch + 1}, loss {l:f}')


print('w: ', net[0].weight.data, '\nb: ', net[0].bias.data)
print('w误差', true_w - net[0].weight.data.reshape(true_w.shape), '\nb误差', true_b - net[0].bias.data)

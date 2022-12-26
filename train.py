import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from utils.ResNet import ResNet18
import numpy as np
import matplotlib.pyplot as plt

# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)


# 不同长度数据，统一为一个标准，倍乘x轴
def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len/y_len
    y_times = [i * times for i in y]
    return y_times


# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 读数据
batch_size = 128
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')
# 加载模型(使用预处理模型，修改最后一层，固定之前的权重)
n_class = 10
model = ResNet18()
"""
ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息,
所以在实验中我们将7x7的降采样层和最大池化层去掉,替换为一个3x3的降采样卷积,
同时减小该卷积层的步长和填充大小
"""
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层改掉
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(device)
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)

# 开始训练
n_epochs = 60
valid_loss_min = np.Inf # track change in validation loss
accuracy = []
lr = 0.1
counter = 0
train_loss2 = []
for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    
    # 动态调整学习率
    if counter/10 ==1:
        counter = 0
        lr = lr*0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ###################
    # 训练集的模型 #
    ###################
    model.train() #作用是启用batch normalization和drop out
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        # clear the gradients of all optimized variables（清除梯度）
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # (正向传递：通过向模型传递输入来计算预测输出)
        output = model(data).to(device)  #（等价于output = model.forward(data).to(device) ）
        # calculate the batch loss（计算损失值）
        loss = criterion(output, target)
        train_loss2.append(loss.item())                     # 损失加入到列表中
        # backward pass: compute gradient of the loss with respect to model parameters
        # （反向传递：计算损失相对于模型参数的梯度）
        loss.backward()
        # perform a single optimization step (parameter update)
        # 执行单个优化步骤（参数更新）
        optimizer.step()
        # update training loss（更新损失）
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # 验证集的模型#
    ######################

    model.eval()  # 验证模型
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)    
        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
    print("Accuracy:",100*right_sample/total_sample,"%")
    accuracy.append(right_sample/total_sample)
 
    # 计算平均损失
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # 显示训练集与验证集的损失函数 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # 如果验证集损失函数减少，就保存模型。
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'checkpoint/resnet18_cifar10.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1
with open("./train_loss.txt", 'w') as train_los:
    train_los.write(str(train_loss2))
train_loss_path = r"./train_loss.txt"   # 存储文件路径

y_train_loss = data_read(train_loss_path)        # loss值，即y轴
x_train_loss = range(len(y_train_loss))			 # loss的数量，即x轴

plt.figure()

    # 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')    # x轴标签
plt.ylabel('loss')     # y轴标签

# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
plt.legend()
plt.title('Loss curve')
plt.show()
plt.savefig("./loss.png")
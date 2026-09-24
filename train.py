import os
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from module import net, cross_entropy, train, DATA_DIR, WORK_DIR
from vis import Animator

#下载数据集
trans = transforms.ToTensor() #使用ToTensor把一张PIL图片或NumPy数组转成PyTorch张量，并自动归一化到[0, 1]
#trans现在是一个"待用"的变换对象，接下来通过torchvision下载数据集，并调用这个trans将数据集中的图片转换为张量形式
mnist_train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, transform=trans, download=True) #训练数据集
mnist_test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, transform=trans, download=True) #测试数据集
#好消息是，torchvision会自动检查目录下是否已经存在所需文件，所以不必担心启动程序时重复下载

#读取数据集
batch_size = 256 #每一批次读取256张图片
dataloader_workers=4 #使用4个进程来读取数据

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=dataloader_workers)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=dataloader_workers)

#初始化参数
num_inputs = 784 #图片是28*28，即有784个像素，输入为784
num_outputs = 10 #分类为0~9中的一个数字，输出为10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) #权重矩阵，初始值标准正态分布服从随机生成
b = torch.zeros(num_outputs, requires_grad=True) #偏置，初始值置0

#超参数
num_epochs = 10 #迭代周期
lr =0.1 #学习率

#训练时逐轮刷新曲线（动画）
animator = Animator() #动画窗口，在train() 中每跑完一轮就会调一次 animator.add()，把当前一轮的数据传进去进行绘制

#进行训练
train(net, train_iter, test_iter, cross_entropy, num_epochs, W, b, lr, history=animator)

#保存训练好的权重与偏置，供 infer.py 推理时读取（路径与 module.load_model 保持一致）
os.makedirs(WORK_DIR, exist_ok=True) #确保保存目录存在
torch.save({"W": W, "b": b}, WORK_DIR / "model.pt")

#训练结束后存一份 PNG（复用动画窗口，不会另开一个），并让窗口留在屏幕上
animator.save(WORK_DIR / "softmax_mnist.png")
animator.show() #阻塞，关掉窗口脚本才结束

#数据流向：train -> train_epoch -> net -> cross_entropy -> accuracy -> evaluate_accuracy
#                      |                     \-> sgd -> 回到train_epoch
#                  animator.add -> 逐轮刷新曲线 -> plot_training_curves -> PNG
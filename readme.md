# 从零实现深度学习手写数字分类

## 前言

* 本项目以李沐老师的《动手学深度学习》（d2l）为基础，实现手写数字识别

* 使用的框架是PyTorch，但是不使用框架中预制的自定义层，而是手动构建Softmax模型

* 使用的手写数字数据集来自MNIST

## 环境配置

我的开发环境在WSL中，使用conda进行环境隔离，仅供参考，也可以在Windows下开发，或者使用venv等，按个人工作习惯即可，先创建好工作目录，依次在目录下执行如下命令：

```bash
conda create -n NC python=3.11 #创建虚拟环境
conda activate NC #进入创建的虚拟环境

conda install pytorch #在虚拟环境中安装pytorch
```

如果需要使用训练过程可视化，观察每一轮训练中的损失变化趋势（这一部分代码写在`vis.py`中，由AI完成，仿制了d2l中的Animator），则需要额外安装如下部分：

```bash
conda install matplotlib
conda install -y --freeze-installed -c conda-forge \
    xorg-libice xorg-libsm xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil
#补充底层库让图像能够显示
```

## 训练过程
* 直接在项目文件夹下运行`python train.py`即可，会自动下载MNIST数据集(若已经下载过了会自动跳过)，并进行训练和测试
* 训练过程会调用`vis.py`的可视化代码，弹窗显示训练过程损失与准确率变化的折线图，并将图片保存到runs/（如果弹出失败，尝试重启WSL）
* 训练结束后把权重`W`和偏置`b`保存成`runs/model.pt`，供接下来推理使用

## 推理过程
* 在项目文件夹下运行`python infer.py`，会读取`train.py`产生的`runs/model.pt`，打开一个手写板，识别你用鼠标写的数字
* 画板操作：**按住鼠标左键写数字 → 回车识别 → 按 c 清空 → 关闭窗口退出**。
* 识别结果会打印在终端里

几点说明：

* 画板本身是黑底白字、28x28，与MNIST的图片格式一致，所以推理时不需要额外的图片预处理
* 模型是单层softmax，真实手写的识别率会明显低于MNIST测试集上的92%左右，写的时候尽量把数字写在中间、写大一点，识别效果最好
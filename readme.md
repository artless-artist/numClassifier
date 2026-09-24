# 从零实现深度学习手写数字分类

## 前言

* 本项目以李沐老师的《动手学深度学习》（d2l）为基础，实现手写数字识别

* 使用的框架是PyTorch，但是不使用框架中预制的自定义层，而是手动构建Softmax模型

* 使用的手写数字数据集来自MNIST

## 创建环境

我的开发环境在WSL中，使用conda进行环境隔离，仅供参考，也可以在Windows下开发，或者使用venv等，按个人工作习惯即可，先创建好工作目录，依次在目录下执行如下命令：

```bash
conda create -n NC python=3.11 #创建虚拟环境
conda activate NC #进入创建的虚拟环境

conda install pytorch #在虚拟环境中安装pytorch
```

如果需要使用训练过程可视化，观察每一轮训练中的损失变化趋势（这一部分写在`vis.py`中），则需要额外安装如下部分：

```bash
conda install matplotlib
conda install -y --freeze-installed -c conda-forge \
    xorg-libice xorg-libsm xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil
#补充底层库让图像能够弹出显示
```

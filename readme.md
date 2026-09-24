# 从零实现深度学习手写数字分类

## 前言

* 本项目以李沐老师的《动手学深度学习》（d2l）为基础，实现手写数字识别

* 使用的框架是PyTorch，但是不使用框架中预制的自定义层，而是手动构建Softmax模型

* 使用的手写数字数据集来自MNIST

## 项目结构

```
numClassifier/
├── train.py            训练脚本：下载数据 → 训练 → 保存权重 → 画训练曲线可视化训练过程
├── module.py           核心模块：模型/损失/精度/优化方法的定义 + 训练/推理过程 + 公共常量
├── drawboard.py        手写板：matplotlib 黑底画板 → 截屏降采样成 28x28 供推理使用
├── infer.py            推理入口：读取 runs/model.pt，打开画板识别数字
├── vis.py              训练过程可视化：Animator 逐轮刷新曲线（可选）
├── readme.md
├── data/               自动下载的 MNIST 数据集（已在 .gitignore 中）
└── runs/               训练产物：model.pt、softmax_mnist.png（已在 .gitignore 中）
```

`module.py`是被其余脚本共同依赖的核心，公共常量只在它里面定义一次，其他文件按需导入，
避免各处写死路径和参数：

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `BASE_DIR` | `module.py`所在目录 | 项目根目录，其余路径都由它推导 |
| `DATA_DIR` | `BASE_DIR/"data"` | MNIST 数据集存放目录 |
| `WORK_DIR` | `BASE_DIR/"runs"` | 训练产物存放目录 |
| `BOARD_SIZE` | 28 | 手写板对应 MNIST 的 28x28 |
| `SCALE` | 10 | 画板放大倍数，28 放大到 280 像素方便鼠标落笔 |
| `DPI` | 100 | 画布分辨率 |
| `LINE_WIDTH` | 14 | 笔画粗细（单位：点），约等于 28x28 下的 2 像素 |

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
* 注意：模型是单层softmax，真实手写的识别率会明显低于MNIST测试集上的92%左右，写的时候尽量把数字写在中间、写大一点，识别效果最好

## 数据流向

```
train.py
├─ train(...)                                    循环 num_epochs 轮
│   ├─ train_epoch(...)                          训练一轮
│   │   └─ 每个 batch：
│   │        net(X, W, b) → softmax → y_hat 前向传播计算得到预测值
│   │        cross_entropy(y_hat, y) → l 计算损失
│   │        l.sum().backward() → sgd([W, b], lr, X.shape[0]) 更新参数
│   │        accuracy(y_hat, y) 计算精度
│   ├─ evaluate_accuracy(net, W, b, test_iter)   完成迭代周期后进行一次整体评估
│   │   └─ accuracy(net(X, W, b), y)
│   └─ history.add(...) → Animator.add(...) → plot_training_curves(...) 将训练表现记录下来绘制曲线以可视化
├─ torch.save(...)                               → runs/model.pt 保存训练结果
└─ animator.save(...)                            → runs/softmax_mnist.png 保持训练过程曲线
```

### 训练：`python train.py`

按顺序，数据是这样被处理的（括号里是被调用函数的形式参数）：

1. `transforms.ToTensor()`：把 MNIST 的 PIL 图片转成取值 [0,1] 的张量
2. `torchvision.datasets.MNIST(root=DATA_DIR, train=True/False, transform=trans, download=True)`：读入训练集/测试集
3. `data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)`：按批打包得到`train_iter`（测试集同理得到`test_iter`）
4. 初始化参数`W`(784,10)、`b`(10,)，并建好曲线窗口`Animator()`
5. `train(net, train_iter, test_iter, cross_entropy, num_epochs, W, b, lr, history=animator)`，每轮做三件事：
   * 训练一轮：`train_epoch(net, train_iter, loss, W, b, lr)`
     * 取一批数据`X`(256,1,28,28)与标签`y`(256,)
     * `net(X, W, b)`：`X`先拍扁成(256,784)，与`W`矩阵相乘，加上偏置`b`，再经`softmax(X)`得到概率`y_hat`(256,10)
     * `cross_entropy(y_hat, y)`：按真实标签取出对应概率、取负对数，得到每个样本的损失`l`(256,)
     * `l.sum().backward()`：反向传播求出梯度
     * `sgd(params=[W, b], lr, batch_size=X.shape[0])`：更新参数并把梯度清零
     * `accuracy(y_hat, y)`：累计预测正确数；函数最后返回该轮的`(训练损失, 训练精度)`
   * 评估一次：`evaluate_accuracy(net, W, b, data_iter=test_iter)`，内部对每批调用`accuracy(net(X, W, b), y)`，返回测试精度
   * 记录一轮：`history.add(epoch+1, 训练损失, 训练精度, 测试精度)`，也就是`Animator.add(epoch, train_loss, train_acc, test_acc)`，逐轮重绘曲线
   * 全部轮次跑完后做断言检查（训练损失<0.5，训练/测试精度在0.7~1之间）
6. `torch.save({"W": W, "b": b}, WORK_DIR/"model.pt")`：把训练好的权重与偏置存盘
7. `animator.save(WORK_DIR/"softmax_mnist.png")`：把窗口里的曲线存成PNG；`animator.show()`阻塞，关掉窗口脚本才结束

### 推理：`python infer.py`

1. `load_model(path=WORK_DIR/"model.pt")`：读回`(W, b)`；文件不存在就打印提示并退出
2. `draw_board(W, b)`：打开画板
3. 鼠标按下/拖动由事件回调`on_press(event)`、`on_move(event)`记录笔画，并用`ax.plot(...)`把每一小段实时画出来
4. 按回车触发`recognize()`：
   * `canvas_to_image(fig)`：先`fig.canvas.draw()`让笔画真正落到画布上，再用`fig.canvas.buffer_rgba()`取像素，按10x10分块取平均降采样，得到(28,28)的灰度张量
   * `predict(image, W, b)`：把(28,28)整理成(1,1,28,28)，调用`net(x, W, b)`做一次前向计算，取概率最大的类别
5. 打印`识别结果：x（置信度 y%）`

训练和推理走的是**同一个`net(X, W, b)`**，区别只在于训练会用梯度把`W`/`b`调好并存盘，推理则把存好的`W`/`b`读回来只做一次前向计算。
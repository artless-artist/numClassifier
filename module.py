import torch

def softmax(X):
    """softmax函数"""
    #将模型输出数值进行处理，使其值为[0,1]且总和为1，作为离散概率分布，softmax(X) = exp(x_i) / Σ_j[exp(x_j)]
    X_exp = torch.exp(X) #先对向量中每个元素求自然指数，保证值非负
    partition = X_exp.sum(1, keepdim=True) #保持列维度，把每一个行包含所有元素加起来，得到所在行的元素总和作为归一化因子
    return X_exp / partition #这里应用了广播机制，把每个元素都除以所在行的总值，得到每个样本的概率分布

def net(X,W,b):
    """softmax回归神经网络的代码表示"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
	#我们要把28*28的图片转为784的向量
	#输入X为一个批次的图片，例如包含256张，那么张量X为(256,1,28,28)，需要转换为(256,784)，每行是256张图片之一的每个像素值
	#W.shape[0]=784（权重矩阵的0维是输入维数，即784），-1为自动计算：256*1*28*28 / 784 =256，自动得到256
	#原本矩阵中的像素通道1被合并进784中了，总而言之就是原本的1*28*28被拍扁进了784中
	#需要注意，这里写成两层括号X.reshape((-1, W.shape[0]))，外侧括号是reshape的调用，而内层括号是一个元组，表示目标形状是(?,784)
	#其实只用单层括号也行，这样做只是为了表示传入的是一个整体
	#然后进行torch.matmul，将reshape后的X与权重矩阵W进行矩阵乘法，把784维输入映射到了10维输出中，得到(256,10)
	#然后用广播机制把偏置b加到256行中
	#最后softmax

def cross_entropy(y_hat, y):
    """损失函数，交叉熵"""
    #衡量预测值y_hat与y的偏差程度，即模型的损失函数，H(y,y_hat)= Σ_i[-(y_i)log(y_hat_i)]
    return - torch.log(y_hat[range(len(y_hat)), y]) 
	#y_hat是预测概率分布，当批次含有256张照片，每张照片给它属于10个类别的预测概率时，形状是(256,10)
	#y是256张照片的真实标签，所以对应正确的类别概率为y_i=100%，对应错误的概率为y_i=0%，交叉熵的公式就可以化简为-log(y_hat_i)
    #负号与log对数运算都很直观，重点是怎么做到用一次找到所有对应的log(y_hat_i)，而非用for循环慢慢找
	#里使用了花式索引技巧：y_hat[range(len(y_hat)), y]，这其含义是取y_hat[i,y[i]]
    #比如说比如说y_hat中样本0的预测概率为[0.1, 0.3, 0.6]，而真实标签是[2,0,2,1]，意思是样本0的真实类别为2，样本1的真实类别为0……
    #这时，我们要取样本0预测成功的概率，显然就是取y_hat[0,2]=0.6，也就是y_hat[i,y[i]]
    #设定i的范围在range(len(y_hat))中，就能一次取遍256行(对应256张照片)

def accuracy(y_hat, y): 
    """计算分类精度"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
	#如果预测值y_hat同时满足：
		#len(y_hat.shape) > 1：至少有两维（排除只有一维，已经是分类标签的情况）
		#y_hat.shape[1] > 1: 第1维的维度数大于1（排除二维但一行只有一个元素(即列向量)的情况，不然接下来的argmax只能取到[0]）
        y_hat = y_hat.argmax(axis=1)
	#排除后的y_hat给出的是概率分布，需要使用argmax沿着第一维(类别)找最大概率对应那个类的索引，把（256,10）的概率分布变成（256,）的分类标签
	#这个判断是为了让函数更通用，既能直接处理模型输出的概率分布，也能处理已经分类完成的预测标签
    cmp = y_hat.type(y.dtype) == y #逐元素比较，先把y与y_hat的类型统一，然后逐一比较，根据比较结果返回布尔量到cmp
    return float(cmp.type(y.dtype).sum()) #然后把布尔量转成1或0，再求和，得到的就是预测成功的个数

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n): #接收参数n，表示要累加的变量数量
        self.data = [0.0] * n #创建一个长度为n的列表，每个元素初始化为 0.0
    def add(self, *args): #*args可变参数列表，接收多个要累加的值
        self.data = [a + float(b) for a, b in zip(self.data, args)]
		#zip(self.data, args)将当前积累值与传入的新值一一配对
		#传入的新值b转为浮点数(保证不同格式兼容)加到已有的积累值a
		#最终生成新列表重新赋给self.data
    def reset(self):
        self.data = [0.0] * len(self.data) #重置，用于下一轮统计
    def __getitem__(self, idx): #让定义的方法可以用下标访问
        return self.data[idx]

def evaluate_accuracy(net, W ,b, data_iter):
    """计算在指定数据集上模型的精度"""
    metric = Accumulator(2)  #使用Accumulator同时累计2个值
    with torch.no_grad(): #因为是评估，不需要计算梯度
        for X, y in data_iter: #X与y是data_iter(到时会传入测试数据集)解包出来的局部名字，此处为遍历整个数据集
            metric.add(accuracy(net(X,W,b), y), y.numel()) #累加计算正确预测数与预测总数
    return metric[0] / metric[1] #分类正确样本数/总样本数

def sgd(params, lr, batch_size): #params是需要更新的参数列表，lr是梯度下降的学习率
    """梯度下降法"""
    with torch.no_grad(): #在下列的块中关闭梯度计算，避免更新参数的操作被保存到计算图中，干扰到反向传播时的计算过程
        for param in params: #遍历每个参数
            param -= lr * param.grad / batch_size #将参数更新，原地操作避免参数内存地址变了下一轮找不到
            param.grad.zero_() #把梯度清零，避免下一轮的梯度累加到上一轮，末尾的下划线表示原地操作

def train_epoch(net, train_iter, loss, W, b, lr):
    """训练模型的一个迭代周期"""
    metric = Accumulator(3) #累加器计算训练损失总和、训练准确度总和、样本数
    for X, y in train_iter: #X与y是train_iter解包出来的局部名字，此处为遍历整个训练集
        #计算梯度并更新参数
        y_hat = net(X, W, b) #正向传播并记录中间值
        l = loss(y_hat, y) #损失函数使用之前的cross_entropy
        l.sum().backward() #进行反向传播，更新参数
        sgd([W, b], lr, X.shape[0]) #已经对输入数据处理过了，此处X.shape[0] = batch_size
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2] #返回训练损失和训练精度

def train(net, train_iter, test_iter, loss, num_epochs, W, b, lr, history=None):#将前面定义的函数传入
    """训练模型的完整过程
    传入 history 时会逐轮记录指标，训练过程本身不受影响
    history 采用鸭子类型，只要有add(轮次, 训练损失, 训练精度, 测试精度) 方法即可，例如 vis.TrainingHistory
    这样本模块不必依赖 matplotlib
    """
    for epoch in range(num_epochs): #训练num_epochs轮
        train_metrics = train_epoch(net, train_iter, loss, W, b, lr) #进行训练
        test_acc = evaluate_accuracy(net, W, b, test_iter) #评估分类精度

        if history is not None: 
            history.add(epoch + 1, train_metrics[0], train_metrics[1], test_acc) 
            #把这一轮的指标喂给训练历史记录器，训练结束后交给 vis.py中的函数画折线图

    train_loss, train_acc = train_metrics #在多轮训练结束后，取出最后一轮的数据
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
	#最后的部分是断言检查：assert 条件 （条件为真则进行执行，假则抛AssertionError，并打印提示信息），这里的三个断言代表：
	#最终训练损失小于 0.5 确认模型收敛
	#训练精度在0.7到1之间 确认训练精度合理
	#测试精度在0.7到1之间 确认泛化正常
	#任意条件不满足，会导致断言失败，也就是提醒训练出现问题
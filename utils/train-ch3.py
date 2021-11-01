import torch
from d2l import torch as d2l
from IPython.core import display


# 定义累加类
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    # 这里是核心操作
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 辅助动画类
class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


# 计算预测正确的个数
def accuracy(y_hat, y):
    # 如果y_hat是矩阵,那么假定第二个维度存储每个类的预测分数,我们使用argmax获得每行中最大元素的索引来获得预测类别
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # 将预测类别与真实y元素进行比较(由于等式运算符==对数据类型很敏感,因此将y_hat的数据类型转换为与y的数据类型一致)
    cmp = y_hat.type(y.dtype) == y
    # 上面比较的结果是一个包含0(错)和1(对)的张量,进行求和便可以得到正确预测的数量
    return float(cmp.type(y.dtype).sum())


# 定义准确率计算
def evaluate_accuracy(net, data_iter):
    # 计算在指定数据集上模型的精度
    # 这里针对不同的输入模型进行处理,增加方法的适用性
    if isinstance(net, torch.nn.Module):
        # 将模型设置为评估模式
        net.eval()

    # metric里存储正确预测数、预测总数
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())

    # 返回正确预测数/预测总数
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期
    :param net: 模型
    :param train_iter: 训练集
    :param loss: 损失函数
    :param updater: 优化算法
    :return: 返回训练损失和训练准确率
    """

    # 下面的代码对于输入的模型和优化算法进行了许多的判断,以增加算法是普适性
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()

    # metric存储训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)

    # 开始本轮训练
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)

        # 根据输入的优化算法来进行不同的操作
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型
    :param net: 模型
    :param train_iter: 训练集
    :param test_iter: 测试集
    :param loss: 损失函数
    :param num_epochs: 训练轮数
    :param updater: 优化算法
    :return:
    """

    # 定义动画类,帮助在训练过程中绘图
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    # 训练
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))

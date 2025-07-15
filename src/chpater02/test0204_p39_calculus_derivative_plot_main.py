# 微积分导数画图
# 解释导数
import numpy as np
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

# 数值函数（当x=1时，导数为2）
def f(x):
    return 3 * x ** 2 - 4 * x

# 使用svg格式绘图
def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')


# set_figsize函数：设置图表大小
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# set_axes()函数： 设置由matplotlib生成图表的轴的属性
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if legend:
        axes.legend(legend)
    axes.grid()
# plot()函数： 简洁绘制曲线
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear'
         , yscale='linear', fmts=('-', 'm--', 'g-', 'r:'), figsize=(3.5, 2.5), axes=None):
    # 绘制数据点
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# 调用plot()函数绘制图像 (Tangent表示切线)
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x -3], 'x', 'f(x)', legend=['f(x)', 'Tangent line(x=1)'])
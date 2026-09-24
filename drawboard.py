import torch
import matplotlib.pyplot as plt
from module import BOARD_SIZE,SCALE,DPI,LINE_WIDTH,predict


def canvas_to_image(fig):
    """把画布上的笔画截屏成 28x28 的灰度图（黑底白字，取值 0~1）。

    matplotlib 画布本身就是一张像素图，直接读它的缓冲区得到一个 (高, 宽, 4) 的张量，
    再按格子取平均降采样到 28x28，就得到了与 MNIST 同样格式的输入。
    """
    fig.canvas.draw()                                    # 确保笔画已经画到画布上
    buffer = fig.canvas.buffer_rgba()                    # 画布的 RGBA 像素，(高, 宽, 4)
    rgba = torch.frombuffer(buffer, dtype=torch.uint8).reshape(buffer.shape)
    gray = rgba[..., :3].float().mean(dim=2)             # 三通道取平均 -> 灰度，0~255

    h, w = gray.shape
    block_h, block_w = h // BOARD_SIZE, w // BOARD_SIZE         # 每个格子占多少像素
    gray = gray[:block_h * BOARD_SIZE, :block_w * BOARD_SIZE]   # 裁成 28 的整数倍
    return gray.reshape(BOARD_SIZE, block_h, BOARD_SIZE, block_w).mean(dim=(1, 3)) / 255.0


def draw_board(W, b):
    """打开手写板：按住鼠标左键写字，回车识别，按 c 清空，关闭窗口退出。"""
    side = BOARD_SIZE * SCALE / DPI                 # 2.8 英寸 * 100 dpi = 280 像素
    fig = plt.figure(figsize=(side, side), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])                 # 坐标轴铺满整张画布，方便整体截屏

    def reset_axes():
        ax.clear()
        ax.set_xlim(0, BOARD_SIZE)
        ax.set_ylim(0, BOARD_SIZE)
        ax.set_facecolor("black")                   # 黑底白字，与 MNIST 一致
        ax.axis("off")

    reset_axes()
    fig.patch.set_facecolor("black")

    strokes = []          # 所有笔画，每条笔画是一串 (x, y) 点
    drawing = False       # 鼠标左键当前是否按下

    def on_press(event):
        nonlocal drawing
        if event.button == 1 and event.xdata is not None:
            drawing = True
            strokes.append([(event.xdata, event.ydata)])

    def on_move(event):
        if not drawing or event.xdata is None:
            return
        x0, y0 = strokes[-1][-1]                    # 上一个点
        x1, y1 = event.xdata, event.ydata           # 当前点
        strokes[-1].append((x1, y1))
        ax.plot([x0, x1], [y0, y1], color="white", linewidth=LINE_WIDTH,
                solid_capstyle="round")             # 把这一段连起来
        fig.canvas.draw_idle()

    def on_release(_event):
        nonlocal drawing
        drawing = False

    def clear(_event=None):#按c则清空画板
        strokes.clear()
        reset_axes()
        fig.canvas.draw_idle()
        print("画板已清空，可以再写一个。")

    def recognize(_event=None):#回车则开始识别
        if not strokes:
            print("画板还是空的，先写一个数字吧。")
            return
        digit, confidence = predict(canvas_to_image(fig), W, b) #进行预测
        message = "识别结果：{}（置信度 {:.1%}）".format(digit, confidence) #产生预测结果
        print(message)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)

    def on_key(event):
        if event.key == "enter":
            recognize()
        elif event.key == "c":
            clear()

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("画板已打开：按住鼠标左键写数字，回车识别，按 c 清空，关闭窗口退出。")
    plt.show()

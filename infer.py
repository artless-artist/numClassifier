from module import load_model #直接使用训练好的模型
from drawboard import draw_board #调用画板程序

model = load_model()
if model is None:
    exit()
W, b = model
draw_board(W, b)

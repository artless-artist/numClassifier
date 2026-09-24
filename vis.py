"""训练过程可视化工具

把每一轮训练的 训练损失 / 训练集分类精度 / 测试集分类精度 画成折线图。

依赖：matplotlib（已装在虚拟环境里；如需补装：pip install matplotlib）

本模块只做「记录 + 绘图」，不改动 module.py 和 train.py 里的任何代码。
但有一点要注意：module.py 的 train() 是在所有轮次跑完之后，才把最后一轮的
train_metrics / test_acc 取出来（见 train() 末尾那两行），所以它拿不到逐轮数据，
直接用它画不出「每一轮」的曲线。要画逐轮曲线，需要在 train.py 里自己写 epoch 循环，
把 train_epoch() 和 evaluate_accuracy() 的返回值逐轮喂给本模块。

============================== 推荐用法（改 train.py，module.py 不用动）==============================

    from module import net, cross_entropy, train_epoch, evaluate_accuracy
    from vis import TrainingHistory, plot_training_curves

    history = TrainingHistory()                     # 1. 建一个记录器

    for epoch in range(num_epochs):                 # 2. 自己写 epoch 循环
        # train_epoch 返回这一轮在训练集上的 (平均损失, 分类精度)
        train_loss, train_acc = train_epoch(net, train_iter, cross_entropy, W, b, lr)
        # evaluate_accuracy 返回这一轮在测试集上的分类精度
        test_acc = evaluate_accuracy(net, W, b, test_iter)

        history.add(epoch + 1,                      # 3. 每轮结束后打一个点
                    train_loss=train_loss,
                    train_acc=train_acc,
                    test_acc=test_acc)

    # 4. 训练结束后出图：左图 = 训练损失折线，右图 = 训练集/测试集精度折线
    history.to_csv("runs/softmax_mnist.csv")        # 落盘，可选
    plot_training_curves(history, save_path="runs/softmax_mnist.png")

等价的省事写法，用 make_epoch_recorder() 直接拿到打点函数::

    from vis import make_epoch_recorder, plot_training_curves

    record, history = make_epoch_recorder()
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, cross_entropy, W, b, lr)
        test_acc = evaluate_accuracy(net, W, b, test_iter)
        record(epoch + 1, train_loss, train_acc, test_acc)  # 轮次, 损失, 训练精度, 测试精度
    plot_training_curves(history)

============================== 其它入口：跳过记录，直接画 ==============================

指标已经存成 CSV 或 dict 时，可以不用 TrainingHistory，直接把数据交给绘图函数::

    # 传 CSV 路径（TrainingHistory.to_csv 存出来的格式）
    plot_training_curves("runs/softmax_mnist.csv", save_path="runs/curve.png")

    # 传 dict，键名固定为 epoch / train_loss / train_acc / test_acc（缺哪个就少画哪条线）
    plot_training_curves({"epoch": [1, 2, 3],
                          "train_loss": [0.9, 0.6, 0.4],
                          "train_acc": [0.75, 0.83, 0.87],
                          "test_acc": [0.72, 0.80, 0.85]})

说明：plot_training_curves(..., show=True) 是默认值，会弹窗显示；只想存图片不弹窗就传
show=False
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

__all__ = [
    "TrainingHistory",
    "plot_training_curves",
    "make_epoch_recorder",
    "setup_matplotlib_style",
]

# --------------------------------------------------------------------------- #
# 中文字体 / 文案配置
# --------------------------------------------------------------------------- #

# 常见的中文字体候选，按优先级排列（Linux / macOS / Windows 都覆盖到）
_CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
]

_LABELS = {
    "zh": {
        "title": "训练过程",
        "xlabel": "训练轮次 (epoch)",
        "loss_ylabel": "损失 (loss)",
        "acc_ylabel": "分类精度 (accuracy)",
        "train_loss": "训练集损失",
        "test_loss": "测试集损失",
        "train_acc": "训练集精度",
        "test_acc": "测试集精度",
    },
    "en": {
        "title": "Training Process",
        "xlabel": "Epoch",
        "loss_ylabel": "Loss",
        "acc_ylabel": "Accuracy",
        "train_loss": "Train Loss",
        "test_loss": "Test Loss",
        "train_acc": "Train Accuracy",
        "test_acc": "Test Accuracy",
    },
}

# 缓存字体探测结果，避免重复扫描系统字体表
_CJK_FONT_NAME = None


def setup_matplotlib_style(font_candidates=None, base_font_size=11):
    """配置 matplotlib 的全局绘图风格，并尝试启用中文字体。

    :param font_candidates: 自定义的中文字体名列表，默认使用内置候选表。
    :param base_font_size: 全局基础字号。
    :return: 是否成功找到一个可用的中文字体（bool）。
    """
    global _CJK_FONT_NAME

    candidates = list(font_candidates) if font_candidates else list(_CJK_FONT_CANDIDATES)

    if _CJK_FONT_NAME is None:
        try:
            installed = {f.name for f in font_manager.fontManager.ttflist}
        except Exception:  # pragma: no cover - 字体表读取失败时退化为英文
            installed = set()
        _CJK_FONT_NAME = next((n for n in candidates if n in installed), "")

    if _CJK_FONT_NAME:
        matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT_NAME, "DejaVu Sans"]
    # 负号用 ASCII 版本，避免中文字体缺失导致的方块
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 120
    matplotlib.rcParams["figure.autolayout"] = False
    matplotlib.rcParams["font.size"] = base_font_size
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.3
    matplotlib.rcParams["grid.linestyle"] = "--"
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = False

    return bool(_CJK_FONT_NAME)


def _resolve_labels(lang="auto", overrides=None):
    """根据语言自动选择文案；没有中文字体时自动回退为英文。"""
    setup_matplotlib_style()

    if lang == "auto":
        lang = "zh" if _CJK_FONT_NAME else "en"

    labels = dict(_LABELS[lang])
    if overrides:
        labels.update(overrides)
    return labels


# --------------------------------------------------------------------------- #
# 指标记录
# --------------------------------------------------------------------------- #

class TrainingHistory:
    """按轮次累积训练指标，并支持 CSV 落盘 / 读回。

    记录的字段（均为可选，缺失的字段绘图时会被自动跳过）：
        - epoch      : 轮次编号
        - train_loss : 该轮在训练集上的平均损失
        - train_acc  : 该轮在训练集上的分类精度
        - test_acc   : 该轮在测试集上的分类精度
        - test_loss  : 该轮在测试集上的平均损失（可选，通常不计算）
    """

    _FIELDS = ("epoch", "train_loss", "train_acc", "test_acc", "test_loss")

    def __init__(self):
        self._data = {name: [] for name in self._FIELDS}

    # ------------------------------ 写入 ------------------------------ #

    def add(self, epoch=None, train_loss=None, train_acc=None,
            test_acc=None, test_loss=None):
        """追加一轮记录。

        :param epoch: 轮次编号；传 None 时自动使用当前已有记录的条数。
        :return: self，方便链式调用。
        """
        if epoch is None:
            epoch = len(self)
        values = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
        }
        # epoch 保持整数形态，其余指标统一为 float（便于直接入图 / 落盘）
        self._data["epoch"].append(_normalize_epoch(values["epoch"]))
        for name in self._FIELDS[1:]:
            self._data[name].append(_to_float_or_none(values[name]))
        return self

    def extend(self, records):
        """批量追加，records 为 dict 的可迭代对象或另一个 TrainingHistory。"""
        if isinstance(records, TrainingHistory):
            for i in range(len(records)):
                self.add(*[records._data[name][i] for name in self._FIELDS])
            return self
        for record in records:
            self.add(**record)
        return self

    def clear(self):
        """清空全部记录，便于复用对象重新训练。"""
        self._data = {name: [] for name in self._FIELDS}

    # ------------------------------ 读取 ------------------------------ #

    @property
    def epochs(self):
        return list(self._data["epoch"])

    @property
    def train_loss(self):
        return list(self._data["train_loss"])

    @property
    def train_acc(self):
        return list(self._data["train_acc"])

    @property
    def test_acc(self):
        return list(self._data["test_acc"])

    @property
    def test_loss(self):
        return list(self._data["test_loss"])

    def to_dict(self):
        """导出为普通 dict（列 -> 列表），方便交给其他绘图库。"""
        return {name: list(values) for name, values in self._data.items()}

    @classmethod
    def from_dict(cls, data):
        """从 dict 构造，缺失的字段按 None 补齐。"""
        history = cls()
        length = max((len(v) for v in data.values()), default=0)
        for i in range(length):
            history.add(**{
                name: _get_item(data.get(name), i)
                for name in cls._FIELDS
            })
        return history

    def __len__(self):
        return len(self._data["epoch"])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            sliced = TrainingHistory()
            for name in self._FIELDS:
                sliced._data[name] = self._data[name][idx]
            return sliced
        return {name: self._data[name][idx] for name in self._FIELDS}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return "TrainingHistory(epochs={}, fields={})".format(
            len(self), [n for n in self._FIELDS[1:] if any(v is not None for v in self._data[n])]
        )

    # --------------------------- 落盘 / 读回 --------------------------- #

    def to_csv(self, path):
        """写入 CSV，自动创建父目录。"""
        _ensure_parent_dir(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self._FIELDS)
            for i in range(len(self)):
                row = [self._data[name][i] for name in self._FIELDS]
                writer.writerow(["" if v is None else v for v in row])
        return path

    @classmethod
    def from_csv(cls, path):
        """从 CSV 读回（与 to_csv 的输出格式对应）。"""
        history = cls()
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                history.add(
                    epoch=_to_float_or_none(row.get("epoch")),
                    train_loss=_to_float_or_none(row.get("train_loss")),
                    train_acc=_to_float_or_none(row.get("train_acc")),
                    test_acc=_to_float_or_none(row.get("test_acc")),
                    test_loss=_to_float_or_none(row.get("test_loss")),
                )
        return history


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #

def plot_training_curves(
    history,
    save_path=None,
    *,
    title=None,
    figsize=(11.0, 4.2),
    lang="auto",
    labels=None,
    show=True,
    marker="o",
    linewidth=1.8,
    markersize=5,
    loss_ylim=None,
    acc_ylim=None,
    annotate_best=True,
    fig=None,
):
    """把每一轮的损失 / 训练集精度 / 测试集精度画成折线图。

    :param history: 支持多种输入，任选其一：
                    - ``TrainingHistory`` 实例
                    - ``dict``（列名 -> 列表，可用 ``TrainingHistory.to_dict()`` 得到）
                    - CSV 文件路径（``str`` / ``os.PathLike``，如 ``TrainingHistory.to_csv`` 的输出）
                    - 由 ``dict`` 组成的可迭代对象，每个 dict 包含 epoch/train_loss/...
    :param save_path: 若提供则把图片保存到该路径（自动创建父目录）。
    :param title: 图标题，默认按语言取"训练过程 / Training Process"。
    :param figsize: 画布尺寸。
    :param lang: ``"auto"`` / ``"zh"`` / ``"en"``，``auto`` 时无中文字体则用英文。
    :param labels: 覆盖默认文案的 dict，键见 ``_LABELS``。
    :param show: 是否调用 ``plt.show()``。无图形界面的服务器建议设为 False。
    :param annotate_best: 是否在测试集精度最高点标注数值。
    :param fig: 传入已有的 Figure 可复用画布；为空则新建。
    :return: ``(fig, (ax_loss, ax_acc))``，方便调用方继续二次定制。
    """
    text = _resolve_labels(lang, labels)

    # ---- 统一入参 ----
    if isinstance(history, TrainingHistory):
        data = history.to_dict()
    elif isinstance(history, dict):
        data = history
    elif isinstance(history, (str, os.PathLike)):
        data = TrainingHistory.from_csv(history).to_dict()
    else:
        data = TrainingHistory().extend(history).to_dict()

    epochs = _clean_series(data.get("epoch"))
    x_default = list(range(1, len(epochs) + 1))
    x = epochs if any(v is not None for v in epochs) else x_default

    train_loss = _clean_series(data.get("train_loss"))
    test_loss = _clean_series(data.get("test_loss"))
    train_acc = _clean_series(data.get("train_acc"))
    test_acc = _clean_series(data.get("test_acc"))

    if not any(s is not None for s in train_loss + test_loss + train_acc + test_acc):
        raise ValueError(
            "history 中没有任何可绘制的指标，请先用 TrainingHistory.add(...) "
            "记录 train_loss / train_acc / test_acc。"
        )

    # ---- 画布 ----
    if fig is None:
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = fig.subplots(1, 2) if not fig.axes else fig.axes
        ax_loss, ax_acc = axes[0], axes[1]

    # ---- 左图：损失 ----
    if any(v is not None for v in train_loss):
        ax_loss.plot(*_pair(x, train_loss), marker=marker, linewidth=linewidth,
                     markersize=markersize, color="#1f77b4", label=text["train_loss"])
    if any(v is not None for v in test_loss):
        ax_loss.plot(*_pair(x, test_loss), marker=marker, linewidth=linewidth,
                     markersize=markersize, color="#ff7f0e", label=text["test_loss"])
    ax_loss.set_xlabel(text["xlabel"])
    ax_loss.set_ylabel(text["loss_ylabel"])
    ax_loss.set_title(_join_labels([
        (text["train_loss"], train_loss),
        (text["test_loss"], test_loss),
    ]), fontsize=11)
    if loss_ylim:
        ax_loss.set_ylim(loss_ylim)
    if any(v is not None for v in train_loss) or any(v is not None for v in test_loss):
        ax_loss.legend(frameon=False)

    # ---- 右图：精度 ----
    if any(v is not None for v in train_acc):
        ax_acc.plot(*_pair(x, train_acc), marker=marker, linewidth=linewidth,
                    markersize=markersize, color="#2ca02c", label=text["train_acc"])
    if any(v is not None for v in test_acc):
        ax_acc.plot(*_pair(x, test_acc), marker=marker, linewidth=linewidth,
                    markersize=markersize, color="#d62728", label=text["test_acc"])
    ax_acc.set_xlabel(text["xlabel"])
    ax_acc.set_ylabel(text["acc_ylabel"])
    ax_acc.set_title(_join_labels([
        (text["train_acc"], train_acc),
        (text["test_acc"], test_acc),
    ]), fontsize=11)
    if acc_ylim:
        ax_acc.set_ylim(acc_ylim)
    if any(v is not None for v in train_acc) or any(v is not None for v in test_acc):
        ax_acc.legend(frameon=False)

    # ---- 最高测试精度标注 ----
    if annotate_best and any(v is not None for v in test_acc):
        bx, by = max(zip(*_pair(x, test_acc)), key=lambda point: point[1])
        ax_acc.annotate(
            "best {:.4f}".format(by),
            xy=(bx, by),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", linewidth=1.0),
        )

    fig.suptitle(title or text["title"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path)

    if show:
        plt.show()

    return fig, (ax_loss, ax_acc)


def make_epoch_recorder(history=None):
    """生成一个"每轮打点"的回调，供训练循环直接调用。

    :param history: 已有的 ``TrainingHistory``；为 None 时内部新建一个。
    :return: ``(recorder, history)``，其中
             ``recorder(epoch, train_loss=None, train_acc=None, test_acc=None, test_loss=None)``
             返回传入的 history 本身。
    """
    history = history if history is not None else TrainingHistory()

    def recorder(epoch, train_loss=None, train_acc=None, test_acc=None, test_loss=None):
        history.add(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                    test_acc=test_acc, test_loss=test_loss)
        return history

    recorder.history = history  # 便于外部取回累积的数据
    return recorder, history


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #

def _to_float_or_none(value):
    """把各种输入统一成 float 或 None（空字符串 / None / NaN 都归为 None）。"""
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # 过滤 NaN


def _normalize_epoch(value):
    """轮次编号：能整除就存成 int，否则存 float；非法值归为 None。"""
    number = _to_float_or_none(value)
    if number is None:
        return None
    return int(number) if number.is_integer() else number


def _join_labels(candidates):
    """把"实际含有数据的曲线名"用 / 连接，作为子图标题。"""
    names = [name for name, series in candidates if any(v is not None for v in series)]
    return " / ".join(names)


def _get_item(seq, idx):
    if seq is None:
        return None
    try:
        return seq[idx]
    except (IndexError, KeyError, TypeError):
        return None


def _clean_series(values):
    """补齐长度并保证每个元素是 float 或 None。"""
    values = list(values) if values is not None else []
    return [_to_float_or_none(v) for v in values]


def _pair(x, ys):
    """把 x 与 y 中"两者都非空"的点配对，跳过缺测轮次。"""
    xs, valid_ys = [], []
    for i, y in enumerate(ys):
        if y is None:
            continue
        xs.append(x[i] if i < len(x) else i + 1)
        valid_ys.append(y)
    return xs, valid_ys


def _ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(str(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)


# --------------------------------------------------------------------------- #
# 自测：直接运行本文件会用模拟曲线生成一张示例图，便于确认环境可用
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # 自测只保存图片、不弹窗，切到 Agg 后端
    matplotlib.use("Agg")

    demo = TrainingHistory()
    for epoch in range(10):
        demo.add(
            epoch=epoch,
            train_loss=1.2 * (0.72 ** epoch) + 0.05,
            train_acc=0.62 + 0.035 * epoch,
            test_acc=0.60 + 0.033 * epoch + (0.01 if epoch % 3 == 0 else 0.0),
        )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    demo.to_csv(os.path.join(out_dir, "demo_history.csv"))
    plot_training_curves(demo, save_path=os.path.join(out_dir, "demo_curves.png"), show=False)
    print("已生成示例：{}/demo_curves.png".format(out_dir))

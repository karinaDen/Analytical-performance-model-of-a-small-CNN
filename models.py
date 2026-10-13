"""Сеть из задания ДЗ-1 (последовательная CNN, без residual).

Слой                                   Выход
Conv7x7 s2 3->32 + ReLU, MaxPool 3x3 s2 p1   32 x S/4 x S/4
Conv5x5 32->64 + ReLU                      64 x S/4
Conv3x3 s2 64->128 + ReLU                 128 x S/8
Conv1x1 128->256 + ReLU                   256 x S/8
Conv3x3 s2 256->256 + ReLU                256 x S/16
Conv1x1 256->512 + ReLU                   512 x S/16
GlobalAvgPool, Linear 512->256, ReLU, Linear 256->100

Все свёртки: padding = k // 2, bias=False. Все ReLU: inplace=True.
BatchNorm в таблице модели нет, поэтому его нет и здесь.

nn.Sequential выбран специально: его forward — это цикл
`for m in self: x = m(x)`, поэтому порядок жизни тензоров известен точно
и пиковую память можно посчитать по формуле (см. equations.py).
"""
import torch
from torch import nn


def conv(cin, cout, k, s=1):
    return nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=k // 2, bias=False)


def build_model(num_classes: int = 100) -> nn.Sequential:
    return nn.Sequential(
        conv(3, 32, 7, 2), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        conv(32, 64, 5), nn.ReLU(inplace=True),
        conv(64, 128, 3, 2), nn.ReLU(inplace=True),
        conv(128, 256, 1), nn.ReLU(inplace=True),
        conv(256, 256, 3, 2), nn.ReLU(inplace=True),
        conv(256, 512, 1), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(512, 256), nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )


if __name__ == "__main__":
    m = build_model()
    print(m)
    print("параметров:", sum(p.numel() for p in m.parameters()))
    with torch.inference_mode():
        for S in (32, 224, 512):
            print(S, tuple(m(torch.randn(2, 3, S, S)).shape))

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        mid = self.mid(self.p4(d4))

        x = self.u4(mid)
        x = self.c4(torch.cat([x, d4], dim=1))
        x = self.u3(x)
        x = self.c3(torch.cat([x, d3], dim=1))
        x = self.u2(x)
        x = self.c2(torch.cat([x, d2], dim=1))
        x = self.u1(x)
        x = self.c1(torch.cat([x, d1], dim=1))
        return self.out(x)


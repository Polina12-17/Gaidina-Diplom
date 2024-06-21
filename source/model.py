import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se="SE"):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        conv_layer = nn.Conv2d
        nlin_layer = nn.LeakyReLU
        if se == "SE":
            SELayer = SEModule
        else:
            SELayer = Identity
        if exp != oup:
            self.conv = nn.Sequential(
                # pw
                conv_layer(inp, exp, 1, 1, 0, bias=True, padding_mode="reflect"),
                nlin_layer(inplace=True),
                # dw
                conv_layer(
                    exp, exp, kernel, stride=stride, padding=padding, groups=exp, bias=True, padding_mode="reflect"
                ),
                SELayer(exp),
                nlin_layer(inplace=True),
                # pw-linear
                conv_layer(exp, oup, 1, 1, 0, bias=True, padding_mode="reflect"),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                conv_layer(inp, exp, 1, 1, 0, bias=True),
                nlin_layer(inplace=False),
                conv_layer(exp, oup, 1, 1, 0, bias=True),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class UnetTMO(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.first_conv = MobileBottleneck(1, 1, 3, 1, 6)
        base_number = 16
        self.conv1 = MobileBottleneck(1, base_number, 3, 2, int(base_number * 1.5), False)
        self.conv2 = MobileBottleneck(base_number, base_number, 3, 1, int(base_number * 1.5), False)
        self.conv3 = MobileBottleneck(base_number, base_number * 2, 3, 2, base_number * 3, False)
        self.conv5 = MobileBottleneck(base_number * 2, base_number * 2, 3, 1, base_number * 3, False)
        self.conv6 = MobileBottleneck(base_number * 2, base_number, 3, 1, base_number * 3, False)
        self.conv7 = MobileBottleneck(base_number * 2, base_number, 3, 1, base_number * 3, False)
        self.conv8 = MobileBottleneck(base_number, 1, 3, 1, int(base_number * 1.5), False)
        self.last_conv = MobileBottleneck(2, 1, 3, 1, 9)

    def forward(self, x):
        x_down = x
        x_1 = self.first_conv(x)
        r = self.conv1(x_1)
        r = self.conv2(r)
        r_d2 = r
        r = self.conv3(r)
        r = self.conv5(r)
        r = self.conv6(r)
        r = F.interpolate(r, (r_d2.shape[2], r_d2.shape[3]), mode="bilinear", align_corners=True)
        r = self.conv7(torch.cat([r_d2, r], dim=1))
        r = self.conv8(r)
        r = F.interpolate(r, (x_down.shape[2], x_down.shape[3]), mode="bilinear", align_corners=True)
        r = self.last_conv(torch.cat([x_1, r], dim=1))
        r = torch.abs(r + 1)
        x = 1 - (1 - x) ** r

        return x, r

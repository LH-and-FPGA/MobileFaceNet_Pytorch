from torch import nn
import torch
import torch.nn.functional as F
import math
import copy
from torch.nn import Parameter


# preLu改ReLU


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1,
                      groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.relu(x)


Mobilefacenet_small_setting = [
    # t, c,  n, s
    [2, 32,  3, 2],
    [2, 64,  1, 2],
    [2, 64,  3, 1],
    [2, 64,  1, 2],
    [2, 64,  1, 1],
]


class MobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_small_setting,
                 inplanes=32, mid_channels=128, embedding_size=64):
        super(MobileFacenet, self).__init__()

        self.conv1 = ConvBlock(3, inplanes, 3, 2, 1)
        self.dw_conv1 = ConvBlock(inplanes, inplanes, 3, 1, 1, dw=True)

        self.inplanes = inplanes
        self.blocks = self._make_layer(Bottleneck, bottleneck_setting)

        last_channel = bottleneck_setting[-1][1]
        self.conv2 = ConvBlock(last_channel, mid_channels, 1, 1, 0)
        self.linear7 = ConvBlock(mid_channels, mid_channels, (7, 6), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(mid_channels, embedding_size, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


def fuse_conv_bn(conv, bn):
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, groups=conv.groups, bias=True,
    )
    gamma = bn.weight.data
    beta = bn.bias.data
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    alpha = gamma / torch.sqrt(var + eps)

    fused_conv.weight.data = conv.weight.data * alpha.view(-1, 1, 1, 1)
    if conv.bias is not None:
        fused_conv.bias.data = alpha * conv.bias.data + beta - alpha * mu
    else:
        fused_conv.bias.data = beta - alpha * mu
    return fused_conv


def fuse_model_bn(model):
    model = copy.deepcopy(model)
    model.eval()

    for _, module in model.named_modules():
        if isinstance(module, ConvBlock):
            fused = fuse_conv_bn(module.conv, module.bn)
            module.conv = fused
            module.bn = nn.Identity()

    for _, module in model.named_modules():
        if isinstance(module, Bottleneck):
            seq = module.conv
            layers = list(seq)
            i = 0
            while i < len(layers) - 1:
                if isinstance(layers[i], nn.Conv2d) and \
                   isinstance(layers[i + 1], nn.BatchNorm2d):
                    layers[i] = fuse_conv_bn(layers[i], layers[i + 1])
                    layers[i + 1] = nn.Identity()
                i += 1
            module.conv = nn.Sequential(*layers)

    return model


def export_onnx(model, filepath="mobilefacenet_pynq.onnx",
                input_size=(1, 3, 112, 96)):
    fused = fuse_model_bn(model)
    fused.eval()
    dummy = torch.randn(*input_size)
    torch.onnx.export(
        fused, dummy, filepath,
        input_names=['input'],
        output_names=['embedding'],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"ONNX exported: {filepath}")
    return filepath


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=64, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


if __name__ == "__main__":
    input_tensor = torch.randn(2, 3, 112, 96)

    net = MobileFacenet()
    total = sum(p.numel() for p in net.parameters())
    print(f"PYNQ model params: {total:,}")

    net.eval()
    with torch.no_grad():
        out = net(input_tensor)
    print(f"Output shape: {out.shape}")

    try:
        export_onnx(net, "mobilefacenet_pynq_fp32.onnx")
    except Exception as e:
        print(f"ONNX export failed: {e}")

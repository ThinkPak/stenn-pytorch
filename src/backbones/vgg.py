import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(
            self,
            input_dim,
            encoder_widths=[16, 32, 64, 128],
            out_widths=[64],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            pad_value=0,
            encoder_norm="group",
            padding_mode="reflect",
            num_heads=8,
            dim_feedforward=128,
            num_layers=1,
            num_classes=20,
            batch_row=1,
    ):
        super(VGG, self).__init__()
        self.batch_row = batch_row
        self.n_stages = len(encoder_widths)
        # 输入卷积块
        self.in_conv = ConvBlock(
            channels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        # 下采样卷积块
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.hidden_dim = sum(encoder_widths)
        # 位置编码
        self.position_embedding = PositionalEncoding(self.hidden_dim)
        # Transform Encoder
        encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出卷积块
        self.out_conv = ConvBlock(
            channels=[self.hidden_dim] + out_widths + [num_classes],
            padding_mode=padding_mode,
        )

    def forward(self, input, batch_positions):
        # 输入卷积块，只改变通道数
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        # 下采样卷积块（三个），通过下采样提取高维特征
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # 先将每个卷积块的特征图上采样到与输入影像尺寸，再在channel维度上进行拼接
        in_h, in_w = input.shape[-2], input.shape[-1]
        out = None
        for feature_map in feature_maps:
            b, t, c, h, w = feature_map.shape
            if in_h != h or in_w != w:
                feature_map = F.interpolate(feature_map.view(-1, c, h, w), size=[in_h, in_w], mode="bilinear")
                feature_map = feature_map.view(b, t, c, in_h, in_w)
            if out is None:
                out = feature_map
            else:
                out = torch.cat((out, feature_map), 2)
        # 位置编码
        out = self.position_embedding(out, batch_positions)
        # 先将特征图按照像素整理成时间序列，再输入到Transformer Encoder中
        result = None
        for obj in out:
            t, c, h, w = obj.shape
            obj = obj.permute((0, 2, 3, 1)).contiguous()
            obj = obj.view((t, -1, c))  # 输入张量形状为(seq, batch, feature)
            for i in range(math.ceil(h / self.batch_row)):
                start = i * self.batch_row
                end = (i + 1) * self.batch_row
                end = end if end < h else h
                obj[:, start * w:end * w, :] = self.transformer_encoder(obj[:, start * w:end * w, :])
            obj = obj.permute((0, 2, 1)).contiguous()
            obj = obj.view((t, c, h, w))
            obj = obj.mean(dim=0).unsqueeze(0)
            obj = self.out_conv(obj)
            if result is None:
                result = obj
            else:
                result = torch.cat((result, obj), dim=0)
        return result


class TemporallySharedBlock(nn.Module):
    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    # smart_forward实现共享空间卷积编码器，并行处理卫星影像时间序列，融合Batch和Time两个维度
    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                            torch.ones(
                                self.out_shape, device=input.device, requires_grad=False
                            )
                            * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    def __init__(self, channels, k=3, p=1, s=1, norm="batch", padding_mode="reflect"):
        super(ConvLayer, self).__init__()
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(num_channels=num_feats, num_groups=4)
        else:
            nl = None
        layers = []
        for i in range(len(channels) - 1):  # 每次循环：Conv2d+GroupNorm+ReLU
            layers.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(channels[i + 1]))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(self, channels, norm="batch", padding_mode='reflect', pad_value=None):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            channels=channels,
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(self, d_in, d_out, k, s, p, norm, padding_mode, pad_value):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(  # 卷积下采样
            channels=[d_in, d_in],
            norm=norm,
            padding_mode=padding_mode,
            k=k,
            s=s,
            p=p,
        )
        self.conv1 = ConvLayer(
            channels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            channels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv3 = ConvLayer(
            channels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # sin and cos position encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, batch_positions):
        position = None
        for obj in batch_positions:
            p = self.pe[obj.tolist()].unsqueeze(0)
            if position is None:
                position = p
            else:
                position = torch.cat((position, p), dim=0)
        position = position.unsqueeze(-1).repeat((1, 1, 1, x.shape[-2])).unsqueeze(-1).repeat((1, 1, 1, 1, x.shape[-1]))
        x = x + position
        return self.dropout(x)

import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from absl import logging,flags
import math
warnings.filterwarnings("ignore", category=UserWarning)

FLAGS = flags.FLAGS

class CALayer(nn.Module):  # Channel Attention (CA) Layer
    def __init__(self, in_channels, reduction=16, pool_types=['avg', 'max']):
        super().__init__()
        self.pool_list = ['avg', 'max']
        self.pool_types = pool_types
        self.in_channels = in_channels
        self.Pool = [nn.AdaptiveAvgPool2d(
            1), nn.AdaptiveMaxPool2d(1, return_indices=False)]
        self.conv_ca = nn.Sequential(
            nn.Conv2d(in_channels, in_channels //
                      reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction,
                      in_channels, 1, padding=0, bias=True)
        )

    def forward(self, x):
        for (i, pool_type) in enumerate(self.pool_types):
            pool = self.Pool[self.pool_list.index(pool_type)](x)
            channel_att_raw = self.conv_ca(pool)
            if i == 0:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw
        scale = F.sigmoid(channel_att_sum)
        return x * scale


class SALayer(nn.Module):  # Spatial Attention Layer
    def __init__(self):
        super().__init__()
        self.conv_sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1, momentum=0.01),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_compress = torch.cat(
            (torch.max(x, 1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)), dim=1)
        scale = self.conv_sa(x_compress)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=2, pool_types=['avg', 'max']):
        super().__init__()
        self.CALayer = CALayer(
            in_channels, reduction, pool_types)
        self.SALayer = SALayer()

    def forward(self, x):
        x_out = self.CALayer(x)
        x_out = self.SALayer(x_out)
        return x_out
    
# class ResDown(nn.Module):
#     """
#     Residual down sampling block for the encoder
#     """

#     def __init__(self, channel_in, channel_out, kernel_size=3 , stride=2, padding=1):
#         super(ResDown, self).__init__()
#         self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, stride, padding)
#         self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
#         self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, 1)
#         self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

#         self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding)

#         self.act_fnc = nn.ELU()

#     def forward(self, x):
#         skip = self.conv3(x)
#         x = self.act_fnc(self.bn1(self.conv1(x)))
#         x = self.conv2(x)
#         return self.act_fnc(self.bn2(x + skip))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.fc1   = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# class ResDown(nn.Module):
#     """
#     Residual down sampling block for the encoder
#     """

#     def __init__(self, channel_in, channel_out, kernel_size=3 , stride=2, padding=1):
#         super(ResDown, self).__init__()
#         self.ChannelAttention = ChannelAttention(channel_out)

#         self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, stride, padding)
#         self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
#         self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, padding)
#         self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

#         self.conv3 = nn.Conv2d(channel_in, channel_out, 1, stride, padding=0)

#         self.act_fnc = nn.ELU()

#     def forward(self, x):
#         skip = self.conv3(x)
#         x = self.act_fnc(self.bn1(self.conv1(x)))
#         x = self.conv2(x)
#         x = self.act_fnc(self.bn2(x + skip))

#         # att_weight = self.ChannelAttention(x)
#         # x = x * att_weight

#         return x
    
class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3 , stride=2, padding=1):
        super(ResDown, self).__init__()
        self.ChannelAttention = ChannelAttention(channel_out)

        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 1, stride, padding=0)

        # self.act_fnc = nn.ELU()
        self.act_fnc = nn.ReLU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.act_fnc(self.bn2(x + skip))

        # att_weight = self.ChannelAttention(x)
        # x = x * att_weight

        return x
    


class Feat_Extractor(nn.Module):
    def __init__(self, channels, hidden=256, latent_channels=1024):
        super(Feat_Extractor, self).__init__()
        self.conv_in = nn.Conv1d(channels, hidden, 3, 2, 1)
        # self.res_down_block1 = ResDown(ch, 2 * ch, 3, (3,2), 1)
        # self.res_down_block2 = ResDown(2 * ch, 4 * ch, 3, (3,2), 1)
        # self.res_down_block3 = ResDown(4 * ch, 8 * ch, 3, (3,2), 1)
        # self.res_down_block4 = ResDown(8 * ch, 16 * ch, 3, (3,2), 1)
        self.res_down_block1 = ResDown(1,  latent_channels//128)
        self.res_down_block2 = ResDown(latent_channels//128, latent_channels//64)
        self.res_down_block3 = ResDown(latent_channels//64, latent_channels//32)
        self.res_down_block4 = ResDown(latent_channels//32, latent_channels//16)
        self.res_down_block5 = ResDown(latent_channels//16, latent_channels//8)
        self.res_down_block6 = ResDown(latent_channels//8, latent_channels//4)
        self.res_down_block7 = ResDown(latent_channels//4, latent_channels//2)
        self.conv_out = nn.Conv2d(latent_channels//2, latent_channels, (2,4), 1)
        # self.act_fnc = nn.ELU()
        self.act_fnc = nn.ReLU()

        
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.act_fnc(self.conv_in(x))
        x = x.unsqueeze(1)
        x = self.res_down_block1(x)  # 32
        x = self.res_down_block2(x)  # 16
        x = self.res_down_block3(x)  # 8
        x = self.res_down_block4(x)  # 4
        x = self.res_down_block5(x)  # 4
        x = self.res_down_block6(x)  # 4
        x = self.res_down_block7(x)  # 4
        x = self.conv_out(x)  # 1

        return x
        

        
# class CovnDown(nn.Module):
#     def __init__(self, channel_in, channel_out, kernel_size=3 , stride=2, padding=1):
#         super(CovnDown, self).__init__()
#         self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 1, padding='same')
#         self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
#         self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, padding='same')
#         self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

#         self.act_fnc = nn.ELU()
#         self.maxpooling = nn.MaxPool2d(kernel_size=(3,2), stride=(3,2), padding=1)

#     def forward(self, x):
#         x = self.act_fnc(self.bn1(self.conv1(x)))
#         x = self.act_fnc(self.bn2(self.conv2(x)))
#         x = self.maxpooling(x)
#         return x


# class Feat_Extractor(nn.Module):
#     def __init__(self, channels, ch=16, latent_channels=128, return_mu_var=True):
#         super(Feat_Extractor, self).__init__()
#         self.conv_in = nn.Conv2d(channels, ch, (7,3), (3,2), (3,1))
#         self.res_down_block1 = CovnDown(ch, 2 * ch, 3, 1, 1)
#         self.res_down_block2 = CovnDown(2 * ch, 4 * ch, 3, 1, 1)
#         self.res_down_block3 = CovnDown(4 * ch, 8 * ch, 3, 1, 1)
#         self.res_down_block4 = CovnDown(8 * ch, 16 * ch, 3, 1, 1)
#         self.conv_mu = nn.Conv2d(16 * ch, latent_channels, (5,5), 1)
#         self.conv_log_var = nn.Conv2d(16 * ch, latent_channels, (5,5), 1)
#         self.act_fnc = nn.ELU()
#         self.return_mu_var = return_mu_var

#     def sample(self, mu, log_var):
#         std = torch.exp(0.5*log_var)
#         eps = torch.randn_like(std)
#         return mu + eps*std
        
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.act_fnc(self.conv_in(x))
#         x = self.res_down_block1(x)  # 32
#         x = self.res_down_block2(x)  # 16
#         x = self.res_down_block3(x)  # 8
#         x = self.res_down_block4(x)  # 4
#         mu = self.conv_mu(x)  # 1
#         log_var = self.conv_log_var(x)  # 1

#         if self.training:
#             x = self.sample(mu, log_var)
#         else:
#             x = mu
        
#         if self.return_mu_var:
#             return x, mu, log_var
#         else:
#             return x
        

class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, padding=1, scale_factor=2, use_padding_layer=False):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, 1, padding)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, 1)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, padding)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.use_padding_layer = use_padding_layer
        if self.use_padding_layer:
            self.padding_layer = nn.ZeroPad2d((0, 0, 0, 1))

        self.conv_out = nn.Conv2d(channel_out, 1, 3, 1, 1)
        # self.conv_out = nn.Conv2d(channel_out, 1, 1, 1, 0)
        self.alpha= nn.Parameter(torch.zeros(1),requires_grad=True)
        # self.act_fnc = nn.ELU()
        self.act_fnc = nn.ReLU()
        self.ChannelAttention = ChannelAttention(channel_out)
        self.CBAM = CBAM(channel_out)


    def forward(self, x, y=None):
        x = self.up_nn(x)
        if self.use_padding_layer:
            x = self.padding_layer(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        # x = self.conv2(x)
        # x = self.act_fnc(self.bn2(x + skip))
        x = self.bn2(self.conv2(x))
        x = self.act_fnc(x + skip)

        # att_weight = self.ChannelAttention(x)
        # x = x * att_weight
        
        # x = self.CBAM(x)

        out = self.CBAM(x)
        out = self.conv_out(out)
        
        if y is not None:
            out = (out + y)/2
            # out = self.alpha*out + (1-self.alpha)*y1
        return x, out
    



class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, latent_channels=512, return_feat=False):
        super(Decoder, self).__init__()
        self.return_feat = return_feat
        self.conv_t_up = nn.ConvTranspose2d(latent_channels, latent_channels//2, (2,5), 1)
        self.res_up_block1 = ResUp(latent_channels//2, latent_channels//4, 3, 1, use_padding_layer=True)
        self.res_up_block2 = ResUp(latent_channels//4, latent_channels//8, 3, 1)
        self.res_up_block3 = ResUp(latent_channels//8, latent_channels//16, 3, 1)
        self.res_up_block4 = ResUp(latent_channels//16, latent_channels//32, 3, 1)
        # self.act_fnc = nn.ELU()
        self.act_fnc = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        if self.return_feat:
            feat0 = self.act_fnc(self.conv_t_up(x))  # 4
            feat1 = self.res_up_block1(feat0)  # 8
            feat2 = self.res_up_block2(feat1)  # 16
            feat3 = self.res_up_block3(feat2)  # 32
            feat4 = self.res_up_block4(feat3)  # 64
            out = self.conv_out(feat4)

            feats = [feat0, feat1, feat2, feat3, feat4]
            return out, feats
        else:
            x = self.act_fnc(self.conv_t_up(x))  # 4
            x, out1 = self.res_up_block1(x)  # 8
            x, out2 = self.res_up_block2(x, self.upsample(out1))  # 16
            x, out3 = self.res_up_block3(x, self.upsample(out2))  # 32
            x, out4 = self.res_up_block4(x, self.upsample(out3))  # 64
            outs = [out1.squeeze(1), out2.squeeze(1), out3.squeeze(1), out4.squeeze(1)]
            # x = torch.tanh(self.conv_out(x))

            return outs

class Feat_Extractor_Decoder(nn.Module):
    def __init__(self, channel_in=1, hidden=256, latent_channels=1024, return_feat=False):
        super(Feat_Extractor_Decoder, self).__init__()
        
        self.return_feat = return_feat
        self.extractor = Feat_Extractor(channel_in, hidden=hidden, latent_channels=latent_channels)
        self.decoder = Decoder(latent_channels=latent_channels, return_feat=self.return_feat)
        

    def forward(self, x):
        encoding = self.extractor(x)
        if self.return_feat:
            recon_img, feats = self.decoder(encoding)
            recon_img = recon_img.squeeze(1)
            return recon_img, encoding, feats
        else:
            recon_imgs = self.decoder(encoding)
            return recon_imgs
        
    
    def decoder_forward(self, encoding):
        if self.return_feat:
            recon_img, feats = self.decoder(encoding)
            recon_img = recon_img.squeeze(1)
            return recon_img, feats
        else:
            recon_imgs = self.decoder(encoding)
            return recon_imgs


        
if __name__ == '__main__':
    device = 'cuda:0'
    model =  Feat_Extractor_Decoder(channel_in=127, hidden=256, latent_channels=256).to(device)
    # x = torch.randn(8, 34, 80).to(device) # batch_size*seq_len*enc_in
    x = torch.randn(8, 1024, 127).to(device) # batch_size*seq_len*enc_in
    recon_imgs = model(x)
    print(recon_imgs[0].shape)
    print(recon_imgs[1].shape)
    print(recon_imgs[2].shape)
    print(recon_imgs[3].shape)
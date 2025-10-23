import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x




#  ema(指数移动平均): 构造趋势（Extracting trend），从原始序列中提取“平滑、缓慢变化的成分”，剔除高频噪声和局部波动。
#  把每个时刻的值，视为过去多个时间点的加权平均
#  越靠近现在的点权重大，越远的点权重小（即：记忆性 & 衰减性）
class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)
        elif ma_type == 'moving_avg':
            self.ma = moving_avg(25, 1)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average
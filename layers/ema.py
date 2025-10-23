import torch
from torch import nn


#  把每个时刻的值，视为过去多个时间点的加权平均
#  越靠近现在的点权重大，越远的点权重小（即：记忆性 & 衰减性）
class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """
    def __init__(self, alpha):
        super(EMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to(x.device)
        divisor = weights.clone()
        #  weights = [0.49, 0.7 * 0.3, 1.0 * 0.3] = [0.49, 0.21, 0.3]
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        #  divisor = [0.49, 0.7, 1.0]
        divisor = divisor.reshape(1, t, 1)
        #  x_ema[0] = x[0] * 0.49 / 0.49 = x[0]
        #  x_ema[1] = (x[0] * 0.49 + x[1] * 0.21) / 0.7
        #  x_ema[2] = (x[0] * 0.49 + x[1] * 0.21 + x[2] * 0.3) / 1.0
        x = torch.cumsum(x * weights, dim=1)  ## 从前往后累加
        x = torch.div(x, divisor)
        return x.to(torch.float32)
    
    # # Naive implementation with O(n) time complexity
    # def forward(self, x):
    #     # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
    #     s = x[:, 0, :]
    #     res = [s.unsqueeze(1)]
    #     for t in range(1, x.shape[1]):
    #         xt = x[:, t, :]
    #         s = self.alpha * xt + (1 - self.alpha) * s
    #         res.append(s.unsqueeze(1))
    #     return torch.cat(res, dim=1)
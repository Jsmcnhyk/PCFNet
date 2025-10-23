import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def stable_period_selector(x, k=3, min_period=3, max_period=None, stability_weight=0.7):
    """
    FFT-based stable period selection combining amplitude and stability.
    Selects top-k periods that are both strong and stable across batches.
    """
    B, T, C = x.shape
    if max_period is None:
        max_period = T // 2

    # FFT analysis for frequency amplitudes
    xf = torch.fft.rfft(x, dim=1)
    amplitudes = abs(xf).mean(-1)  # [B, freq]

    # Stability scoring based on cross-batch variance
    mean_amp = amplitudes.mean(0)
    var_amp = amplitudes.var(0) + 1e-8
    stability_score = 1.0 / var_amp

    # Combined scoring: amplitude * stability
    score = mean_amp * (stability_score ** stability_weight)

    # Filter valid periods and select top-k
    score[0] = 0  # Remove DC component
    freqs = torch.arange(len(score))
    periods = T / freqs.clamp(min=1)
    valid_mask = (periods >= min_period) & (periods <= max_period)
    score[~valid_mask] = -1

    top_values, top_idx = torch.topk(score, k)
    best_periods = (T / top_idx.clamp(min=1)).int().cpu().numpy()

    selected_amplitudes = amplitudes[:, top_idx]

    return best_periods, selected_amplitudes


class StablePeriodEncoder(nn.Module):
    """Period-aware encoder using separable periodic convolution"""

    def __init__(self, configs, mark_c):
        super(StablePeriodEncoder, self).__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model

        self.period_channel_processor = nn.Sequential(
            SeparablePeriodicConv(configs.enc_in + mark_c, configs.enc_in + mark_c,
                                  num_kernels=configs.num_kernels),
        )

    def forward(self, x, period):
        x = x.permute(0, 2, 1)  # [B, C, T]
        B, T, N = x.size()

        # Zero-pad for period divisibility
        if self.d_model % period != 0:
            length = ((self.d_model // period) + 1) * period
            padding = torch.zeros([B, (length - self.d_model), N], device=x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = self.d_model
            out = x

        # Reshape to 2D period structure: [batch, channels, period_num, period_len]
        out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

        # Apply separable convolution
        out = self.period_channel_processor(out)

        # Reshape back and trim padding
        out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
        out = out[:, :self.d_model, :]

        return out.permute(0, 2, 1)


class SeparablePeriodicConv(nn.Module):
    """Separable convolution for intra-period and inter-period patterns"""

    def __init__(self, in_channels, out_channels, num_kernels=3):
        super(SeparablePeriodicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        # Temporal convolutions for intra-period patterns
        self.temporal_convs = nn.ModuleList()
        for i in range(self.num_kernels // 2):
            kernel_size = 2 * i + 3
            self.temporal_convs.append(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=[1, kernel_size],
                          padding=[0, kernel_size // 2],
                          groups=in_channels)
            )

        # Period convolutions for inter-period patterns
        self.period_convs = nn.ModuleList()
        for i in range(self.num_kernels // 2):
            kernel_size = 2 * i + 3
            self.period_convs.append(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=[kernel_size, 1],
                          padding=[kernel_size // 2, 0],
                          groups=in_channels)
            )

        # Pointwise mixing for feature integration
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1, groups=in_channels)

        # Learnable fusion weights
        total_branches = len(self.temporal_convs) + len(self.period_convs) + 1
        self.weights = nn.Parameter(torch.ones(total_branches))

    def forward(self, x):
        outputs = []

        # Process temporal patterns (within periods)
        for conv in self.temporal_convs:
            outputs.append(conv(x))

        # Process period patterns (across periods)
        for conv in self.period_convs:
            outputs.append(conv(x))

        # Add pointwise features
        outputs.append(self.pointwise_conv(x))

        # Adaptive weighted fusion
        weights = F.softmax(self.weights, dim=0)
        result = sum(w * out for w, out in zip(weights, outputs))

        return result


class PeriodMaskProcessor(nn.Module):
    """Generate masked versions by blocking different period chunks"""

    def __init__(self, d_model, factor):
        super(PeriodMaskProcessor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * factor)),
            nn.GELU(),
            nn.Linear(int(d_model * factor), d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x, period):
        results = []
        seq_len = x.size(-1)
        chunk_len = seq_len // period

        # Generate masked versions by blocking each period chunk
        for i in range(period):
            masked_x = x.clone()
            start_idx = i * chunk_len
            end_idx = (i + 1) * chunk_len
            masked_x[..., start_idx:end_idx] = 0
            results.append(self.mlp(masked_x))

        # Handle remainder if sequence not divisible by period
        if seq_len % period != 0:
            masked_x = x.clone()
            masked_x[..., period * chunk_len:] = 0
            results.append(self.mlp(masked_x))

        return results


class PeriodMaskFusion(nn.Module):
    """Adaptive fusion of period-masked features for complementary learning"""

    def __init__(self, d_model, factor):
        super(PeriodMaskFusion, self).__init__()
        self.d_model = d_model
        max_results = d_model // 2

        self.periodic_processor = PeriodMaskProcessor(d_model, factor)
        self.weights = nn.Parameter(torch.ones(max_results))
        self.ln = nn.LayerNorm(d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def weighted_sum_results(self, results):
        """Adaptively weight and sum masked results"""
        num_results = len(results)
        active_weights = self.weights[:num_results]
        normalized_weights = F.softmax(active_weights, dim=0)

        weighted_sum = torch.zeros_like(results[0])
        for i, result in enumerate(results):
            weighted_sum += normalized_weights[i] * result

        return weighted_sum

    def forward(self, x, period):
        # Generate complementary masked features
        results = self.periodic_processor(x, period)

        # Adaptive fusion with residual connection
        weighted_v = self.weighted_sum_results(results)
        weighted_v = weighted_v + x
        weighted_v = self.ln(weighted_v)

        return self.v_proj(weighted_v)


class PeriodChannelAttention(nn.Module):
    """Period-aware cross-channel attention for global integration"""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(PeriodChannelAttention, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        # Feed-forward network
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu


    def forward(self, x, period_queries, period_values, period_weight):
        # Stack multi-period features
        qks = torch.stack(period_queries, dim=-1)  # [B, C, T, K]
        vs = torch.stack(period_values, dim=-1)  # [B, C, T, K]

        B, C, T, K = qks.shape

        # Apply period importance weights
        period_weight = F.normalize(period_weight, p=2, dim=1)
        period_weight = F.softmax(period_weight, dim=1)  # [B, K]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, C, T, 1)


        qks = qks * period_weight
        qks = qks.permute(0, 3, 1, 2).reshape(B , K*C, T)
        vs = vs.permute(0, 3, 1, 2).reshape(B , K*C, T)

        new_x = self.attention[0](qks, qks, vs)[0]
        new_x = new_x.reshape(B, K, C, T).permute(0, 2, 3, 1)
        new_x = torch.mean(new_x, dim=-1)


        # Feed-forward with residual connection
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class HybridTrendNet(nn.Module):
    """Hybrid trend prediction with polynomial baseline + neural residual"""

    def __init__(self, input_dim, pred_len, hidden_dim=None, coef_scale=0.1):
        super(HybridTrendNet, self).__init__()
        self.pred_len = pred_len
        if hidden_dim is None:
            hidden_dim = input_dim * 2
        self.coef_scale = coef_scale

        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2)
        )

        # Polynomial coefficients predictor (a + bt + ct^2)
        self.trend_predictor = nn.Linear(hidden_dim // 2, 3)

        # Neural residual correction
        self.residual_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, pred_len * 2),
            nn.GELU(),
            nn.LayerNorm(pred_len * 2),
            nn.Dropout(0.1),
            nn.Linear(pred_len * 2, pred_len)
        )

        # Adaptive balance between polynomial and neural components
        self.adaptive_weight = nn.Sequential(
            nn.Linear(hidden_dim // 2, pred_len),
            nn.Sigmoid()
        )

    def forward(self, t):
        """
        Hybrid trend prediction combining polynomial baseline with neural residual

        Args:
            t: [B, C, L] trend component
        Returns:
            [B, C, pred_len] trend forecast
        """
        batch_size, num_channels, seq_len = t.shape
        t_reshaped = t.reshape(-1, seq_len)

        # Extract trend features
        features = self.feature_extractor(t_reshaped)
        if len(features.shape) == 3:
            global_features = features.mean(dim=1)
        else:
            global_features = features

        # Generate polynomial coefficients
        trend_coeffs = torch.tanh(self.trend_predictor(global_features)) * self.coef_scale

        # Construct polynomial basis [1, t, t^2]
        time_base = torch.linspace(0, 1, self.pred_len, device=t.device)
        poly_basis = torch.stack([torch.ones_like(time_base), time_base, time_base ** 2], dim=-1)

        # Polynomial baseline prediction
        trend_baseline = torch.matmul(poly_basis, trend_coeffs.unsqueeze(-1)).squeeze(-1)

        # Anchor to last observed value for continuity
        last_value = t_reshaped[:, -1:].detach()
        anchored_trend = trend_baseline + last_value

        # Neural residual correction
        residual_correction = self.residual_predictor(global_features)

        # Adaptive combination of polynomial and neural components
        adaptive_weights = self.adaptive_weight(global_features)
        final_pred_flat = anchored_trend + adaptive_weights * residual_correction

        return final_pred_flat.reshape(batch_size, num_channels, self.pred_len)
import argparse
import time

import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(2025)

    parser = argparse.ArgumentParser(description='PerciNet: Period-aware Channel Independent Networks for Long-term Time Series Forecasting')


    # Visualization
    parser.add_argument('--enable_visual', action='store_true', default=False,
                        help='Enable visualization during testing for result analysis and periodic pattern inspection')

    # Core Model Architecture Parameters
    parser.add_argument('--fusion_layers', type=int, default=2,
                        help='Number of value generator layers in MultiPeriodFusion module, controls fusion complexity of periodic features')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Dimension of model embeddings and hidden states, determines DataEmbedding_inverted output size and model capacity')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads in CrossPeriodAttention module for capturing cross-period relationships')
    parser.add_argument('--attn_dropout', type=float, default=0.15,
                        help='Dropout rate for attention mechanisms in CrossPeriodAttention, prevents attention overfitting')
    parser.add_argument('--d_ff', type=int, default=None,
                        help='Dimension of feed-forward networks in fusion layers, auto-set to d_model*4 if None')

    # Period-aware Components Parameters
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top periods selected by stable_period_selector via FFT analysis, balances coverage vs complexity')
    parser.add_argument('--num_kernels', type=int, default=0,
                        help='Number of convolution kernels in SeparablePeriodicConv module for intra/inter-period pattern capture')
    parser.add_argument('--encoder_layers', type=int, default=2,
                        help='Number of StablePeriodEncoder layers for each selected period, controls period-specific feature extraction depth')

    # Trend Processing Parameters
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension of HybridTrendNet feature extractor, auto-set to input_dim*2 if None')
    parser.add_argument('--coef_scale', type=float, default=0.1,
                        help='Coefficient scaling factor for polynomial trend prediction in HybridTrendNet, prevents excessive trend magnitude')

    # Adaptive Components Parameters
    parser.add_argument('--tfactor', type=float, default=1,
                        help='Expansion factor in PeriodMaskEncoder MLP within MultiPeriodFusion, controls nonlinear transformation capacity')

    # Normalization and Decomposition Parameters
    parser.add_argument('--revin', type=int, default=1,
                        help='Enable RevIN reversible normalization for distribution shift handling (1: True, 0: False)')
    parser.add_argument('--ma_type', type=str, default='ema',
                        help='Decomposition type in DECOMP module: reg (regular MA), ema (exponential MA), dema (double EMA)')
    parser.add_argument('--ema_a', type=float, default=0.3,
                        help='Smoothing parameter for EMA trend component in DECOMP (0 < ema_a < 1), higher values = more smoothing')
    parser.add_argument('--ema_b', type=float, default=0.3,
                        help='Smoothing parameter for EMA seasonal component or DEMA in DECOMP (0 < ema_b < 1)')

    # Regularization Parameters
    parser.add_argument('--alpha', type=float, default=0.2, help='Weight of time-frequency domain MAE loss component')
    parser.add_argument('--dropout', type=float, default=0.1, help='General dropout rate for embedding and other components')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training input data')

    # Basic Configuration
    parser.add_argument('--is_training', type=int, required=True, default=1, help='Training status: 1 for training, 0 for testing only')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='Unique identifier for model experiment')
    parser.add_argument('--model', type=str, required=True, default='PerciNet', help='Model architecture name')

    # Data Loader Configuration
    parser.add_argument('--data', type=str, required=True, default='custom', help='Dataset type identifier')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='Root directory path containing data files')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='Specific data CSV filename')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting task type: M (multivariate->multivariate), S (univariate->univariate), MS (multivariate->univariate)')
    parser.add_argument('--target', type=str, default='OT', help='Target feature column name for S or MS forecasting tasks')
    parser.add_argument('--freq', type=str, default='h',
                        help='Time frequency for feature encoding: s (secondly), t (minutely), h (hourly), d (daily), b (business), w (weekly), m (monthly)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Directory path for saving/loading model checkpoints')

    # Forecasting Task Configuration
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length for historical data window')
    parser.add_argument('--label_len', type=int, default=48, help='Start token length (legacy parameter, not used in inverted Transformers)')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction sequence length for future forecasting horizon')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='Seasonal pattern subset specification for M4 dataset')

    # Model Definition Parameters
    parser.add_argument('--enc_in', type=int, default=7, help='Number of input channels/features for encoder')
    parser.add_argument('--embed', type=str, default='timeF', help='Time features encoding method: timeF (time features), fixed, learned')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function used throughout the model')

    # Optimization Parameters
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading')
    parser.add_argument('--itr', type=int, default=1, help='Number of independent experiment iterations for statistical reliability')
    parser.add_argument('--train_epochs', type=int, default=10, help='Total number of training epochs')
    parser.add_argument('--embedding_epochs', type=int, default=5, help='Number of epochs for embedding pre-training')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience: epochs to wait before stopping')
    parser.add_argument('--pct_start', type=float, default=0.2, help='Percentage of training for learning rate warmup')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Main optimizer learning rate')
    parser.add_argument('--embedding_lr', type=float, default=0.0005, help='Learning rate specifically for embedding layers')
    parser.add_argument('--des', type=str, default='test', help='Experiment description for identification and logging')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function type for training')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment schedule type')

    # GPU Configuration
    parser.add_argument('--use_gpu', type=bool, default=True, help='Enable GPU acceleration if available')
    parser.add_argument('--gpu', type=int, default=0, help='Primary GPU device ID for single-GPU training')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Enable multi-GPU distributed training', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='Comma-separated list of GPU device IDs for multi-GPU setup')
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='Enable DTW (Dynamic Time Warping) metric evaluation (computationally expensive, use sparingly)')

    parser.add_argument('--inverse', action='store_true', help='Apply inverse transformation to output predictions', default=False)


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_bs{}_ft{}_sl{}_ll{}_pre{}_dm{}_nh{}_dff{}_tk{}_nk{}_el{}_fl{}_hd{}_cs{}_tf{}_dp{}_adr{}_em{}_lr{}_ep{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.batch_size,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,  # Model dimension (embedding size)
                args.n_heads,  # Number of attention heads
                args.d_ff,  # Feed-forward network dimension
                args.top_k,  # Top-k periods selected by FFT
                args.num_kernels,  # Number of convolutional kernels
                args.encoder_layers,  # Number of period encoding layers
                args.fusion_layers,  # Number of value generator layers
                args.hidden_dim,  # Hidden dimension for trend network
                args.coef_scale,  # Coefficient scaling factor for polynomial trend
                args.tfactor,  # Expansion factor for adaptive components
                str(args.dropout).replace('.', ''),  # Dropout rate
                str(args.attn_dropout).replace('.', ''),  # Attention dropout rate
                args.ema_a,  # EMA alpha parameter for decomposition
                str(args.learning_rate).replace('.', ''),  # Learning rate
                args.train_epochs,  # Number of training epochs
                args.des,  # Experiment description
                ii  # Trial number
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_bs{}_ft{}_sl{}_ll{}_pre{}_dm{}_nh{}_dff{}_tk{}_nk{}_el{}_fl{}_hd{}_cs{}_tf{}_dp{}_adr{}_em{}_lr{}_ep{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.batch_size,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,  # Model dimension (embedding size)
            args.n_heads,  # Number of attention heads
            args.d_ff,  # Feed-forward network dimension
            args.top_k,  # Top-k periods selected by FFT
            args.num_kernels,  # Number of convolutional kernels
            args.encoder_layers,  # Number of period encoding layers
            args.fusion_layers,  # Number of value generator layers
            args.hidden_dim,  # Hidden dimension for trend network
            args.coef_scale,  # Coefficient scaling factor for polynomial trend
            args.tfactor,  # Expansion factor for adaptive components
            str(args.dropout).replace('.', ''),  # Dropout rate
            str(args.attn_dropout).replace('.', ''),  # Attention dropout rate
            args.ema_a,  # EMA alpha parameter for decomposition
            str(args.learning_rate).replace('.', ''),  # Learning rate
            args.train_epochs,  # Number of training epochs
            args.des,  # Experiment description
            ii  # Trial number
        )
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        start_time = time.time()
        exp.test(setting, test=1)
        end_time = time.time()
        print(f"Runtime: {end_time - start_time:.4f} s")
        torch.cuda.empty_cache()
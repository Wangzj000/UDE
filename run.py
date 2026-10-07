import argparse
import os
import torch
from exp.exp_large import Exp_pretrain
from exp.exp_finetune import Exp_Finetune
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Universal Delay Embedding (UDE) - Time Series Foundation Model')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='finetune',
                        help='pretrain, finetune]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of saving checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')



    # model define

    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='1: channel independence')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0, help='early stopping min_delta')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    # my_model hyper-params
    parser.add_argument('--pe', type=str, help='options=[learnable_pe, fix_pe, None]')
    parser.add_argument('--project', type=str, help='options=[conv, linear]', default='conv')
    parser.add_argument('--n_vars', type=int, default=0)
    parser.add_argument('--individual', type=bool, default=True)
    parser.add_argument('--L', type=int, default=8)
    parser.add_argument('--p1', type=int, default=8)
    parser.add_argument('--p2', type=int, default=8)
    parser.add_argument('--pooling_kernel', type=int, default=4)
    parser.add_argument('--pooling_type', type=str, help='options=[max, avg]', default='max')
    
    # pretrain
    parser.add_argument('--train_dataset_list', type=str)
    parser.add_argument('--test_dataset_list', type=str)
    parser.add_argument('--vali_dataset_list', type=str)
    
    # finetune params
    parser.add_argument('--model_location', type=str, help='location of the pretrained model')
    parser.add_argument('--sampling_rate', type=float, default=1.0)
    parser.add_argument('--fc_only', action='store_true')
    parser.add_argument('--fc_num', type=int)
    
    # resume
    parser.add_argument('--resume', action='store_true', help='resume training from the latest checkpoint', default=False)
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to the checkpoint file')
    
    # other models
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'pretrain':
        Exp = Exp_pretrain
    elif args.task_name == 'finetune':
        Exp = Exp_Finetune
    else:
        raise ValueError(f"Unsupported task_name: {args.task_name}. Options: [pretrain, finetune, long_term_forecast]")
        
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.des,
            ii
        )
            # 如果选择恢复训练并提供了检查点路径，则加载检查点
            if args.resume and args.checkpoint_path:
                exp._load_checkpoint(args.checkpoint_path)
                
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.des,
            ii
        )

        exp = Exp(args)  # set experiments
        print(exp)
        # 如果选择恢复训练并提供了检查点路径，则加载检查点
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

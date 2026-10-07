from data_provider.data_loader import Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Solar, UnivariateDatasetBenchmark, MultivariateDatasetBenchmark, UTSD_Npy, UnivariateDatasetBenchmark_finetune, MultivariateDatasetBenchmark_finetune, Climate_univariate, Climate_Multivariate
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader


data_dict = {
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'Solar': Dataset_Solar,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    "UTSD_Npy": UTSD_Npy,
    "UnivariateDatasetBenchmark_finetune": UnivariateDatasetBenchmark_finetune,
    "MultivariateDatasetBenchmark_finetune":MultivariateDatasetBenchmark_finetune,
    'Climate_univariate': Climate_univariate,
    'Climate_Multivariate': Climate_Multivariate
}

 
def data_provider(args, flag, pretrain_flag='finetune'):
    Data = data_dict[args.data]

    if args.task_name != 'finetune':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.seq_len, args.pred_len],
            pretrain_flag=pretrain_flag
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.seq_len, args.pred_len],
            pretrain_flag=pretrain_flag,
            sampling_rate=args.sampling_rate
        )
    print(flag, len(data_set))
    return data_set

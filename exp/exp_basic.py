import os
import torch
from models import delayformer, delayformer_fixpooling, iTransformer, DLinear, PatchTST, wavenet, deepar

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'delayformer': delayformer, 
            'delayformer_fixpooling': delayformer_fixpooling,
            'iTransformer': iTransformer,
            'PatchTST': PatchTST,
            'DLinear': DLinear,
            'wavenet': wavenet,
            'deepar': deepar,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        # print(self.model.state_dict())
        if args.task_name == 'finetune':
            state_dict = torch.load(self.args.model_location, map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                
            if args.data in ['Climate_univariate', 'Climate_Multivariate', 'UnivariateDatasetBenchmark_finetune']:
                keys_to_remove = ['mlp_head', 'pred_dim_change']
                if args.pred_len != 96:
                    keys_to_remove += ['module.linears']
                if args.data == 'Climate_Multivariate':
                    keys_to_remove += ['revin_layer']
                
                # 删除不需要的 key
                keys_to_remove_set = set(k for k in state_dict if any(r in k for r in keys_to_remove))
                for k in keys_to_remove_set:
                    del state_dict[k]

                # 如果不是多卡训练，移除 "module." 前缀
                if not self.args.use_multi_gpu:
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # 对比模型参数
                model_dict = self.model.state_dict()
                matched_keys = model_dict.keys() & state_dict.keys()
                print(f"Matched parameters: {len(matched_keys)}\n")
                for key in sorted(matched_keys):
                    model_shape = model_dict[key].shape
                    load_shape = state_dict[key].shape
                    status = "OK" if model_shape == load_shape else "MISMATCH"
                    print(f"{key:50} Model: {model_shape} | Loaded: {load_shape} --> {status}")

                load_info = self.model.load_state_dict(state_dict, strict=False)
                print(f"Missing keys: {load_info.missing_keys}")

            else:
                if self.args.use_multi_gpu:
                    self.model.load_state_dict(state_dict, strict=True)
                else:
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    self.model.load_state_dict(state_dict, strict=True)
            print("====================Load Pretrained model successfully==========================")
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.start_epoch = 0  # 用于记录恢复训练时的起始 epoch
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)

    def _build_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def _save_checkpoint(self, epoch, save_path):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args
        }
        torch.save(state, save_path)
        print(f"Checkpoint saved at epoch {epoch} to {save_path}")

    def _load_checkpoint(self, load_path):
        if os.path.exists(load_path):
            state_dict = torch.load(load_path)
            new_state_dict = {}
            for k, v in state_dict['model_state_dict'].items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict, strict=True)
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.start_epoch = state_dict['epoch'] + 1  # 从下一个epoch开始
            print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

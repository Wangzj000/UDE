from data_provider.data_factory import data_provider
from torch.utils.data import DataLoader
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json

warnings.filterwarnings('ignore')


class Exp_pretrain(Exp_Basic):
    def __init__(self, args):
        super(Exp_pretrain, self).__init__(args)
        # 初始化训练和验证数据加载器
        self.train_loader = None
        self.vali_loader = None
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, pretrain_flag):
        data_set = data_provider(self.args, flag, pretrain_flag)
        return data_set

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        iter_count = 0
        time_now = time.time()
        skip_dataset = False  # Flag to skip the dataset if NaN is encountered
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                # convert to one-dim
                # batch_size, seq_len, dims = batch_x.shape
                # batch_x = batch_x.permute(0, 2, 1).contiguous().view(batch_size * dims, seq_len, 1)
                # batch_size, seq_len, dims = batch_y.shape
                # batch_y = batch_y.permute(0, 2, 1).contiguous().view(batch_size * dims, seq_len, 1)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                try:
                    # Encoder - Decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = outputs[0] if self.args.output_attention else outputs
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[0] if self.args.output_attention else outputs

                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                    loss = criterion(outputs, batch_y)

                    # if torch.isnan(loss).any().item():
                    #     print("NaN loss detected during validation. Skipping this validation dataset.")
                    #     skip_dataset = True
                    #     break  # Exit the batch loop for this dataset

                    total_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        print("\titers: {}\t \tspeed: {:.4f}s/iter".format(i + 1, speed))
                        iter_count = 0
                        time_now = time.time()
                except Exception as e:
                    print(f"Exception during validation: {e}. Skipping this validation dataset.")
                    skip_dataset = True
                    break  # Exit the batch loop for this dataset
            
        if not skip_dataset:
            total_loss = np.average(total_loss)
        else:
            total_loss = float('nan')  # Assign NaN to indicate skipping
        self.model.train()
        return total_loss

    def train(self, setting):
        if self.args.train_dataset_list == 'json':
            with open('train_data_list.json', 'r') as file:
                train_data_name = json.load(file)
            train_dataset_list = [i for i in train_data_name['files']]
            train_datasets = {}
            print('Training dataset list:')
            for i, data_path in enumerate(train_dataset_list):
                print(data_path)
                self.args.data_path = data_path
                train_datasets[f'dataset{i+1}'] = self._get_data(flag='train', pretrain_flag='pre_training')
        else:
            train_dataset_list = self.args.train_dataset_list.replace(',', ' ').split()
            train_datasets = {}
            for i in range(len(train_dataset_list)):
                if train_dataset_list[i] in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
                    self.args.data_path = 'ETT-small/' + train_dataset_list[i] + '.csv'

                else:
                    self.args.data_path = train_dataset_list[i] + '/' + train_dataset_list[i] + '.csv'
                train_datasets[f'dataset{i+1}'] = self._get_data(flag='train', pretrain_flag='pre_training')
        print(f'Load {len(train_dataset_list)} training datasets successfully.')
        
        # vali_dataset_list = self.args.train_dataset_list
        # vali_dataset_list = self.args.vali_dataset_list
        if self.args.vali_dataset_list == 'json':
            with open('vali_data_list.json', 'r') as file:
                val_data_name = json.load(file)
            vali_dataset_list = [i for i in val_data_name['files']]
            vali_datasets = {}
            print('Validation dataset list:')
            for i, data_path in enumerate(vali_dataset_list):
                print(data_path)
                self.args.data_path = data_path
                vali_datasets[f'dataset{i+1}'] = self._get_data(flag='train', pretrain_flag='pre_training')
        else:
            vali_dataset_list = self.args.train_dataset_list
            vali_dataset_list = vali_dataset_list.replace(',', ' ').split()
            vali_datasets = {}
            for i in range(len(vali_dataset_list)):
                if vali_dataset_list[i] in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
                    self.args.data_path = 'ETT-small/' + vali_dataset_list[i] + '.csv'
                else:
                    self.args.data_path = vali_dataset_list[i] + '/' + vali_dataset_list[i] + '.csv'
                vali_datasets[f'dataset{i+1}'] = self._get_data(flag='vali', pretrain_flag='pre_training')
        print(f'Load {len(vali_dataset_list)} validation datasets successfully.')
        
        self.args.data = 'UnivariateDatasetBenchmark'
        test_dataset_list = self.args.test_dataset_list.replace(',', ' ').split()
        test_datasets = {}
        for i in range(len(test_dataset_list)):
            if test_dataset_list[i] in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
                self.args.data_path = 'ETT-small/' + test_dataset_list[i] + '.csv'
            else:
                self.args.data_path = test_dataset_list[i] + '/' + test_dataset_list[i] + '.csv'
            test_datasets[f'dataset{i+1}'] = self._get_data(flag='test', pretrain_flag='fine_tuning')
        print(f'Load {len(test_dataset_list)} test datasets successfully.')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=self.args.min_delta)

        model_optim = self.optimizer  # 使用从 Exp_Basic 初始化的优化器
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            total_train_loss = []

            self.model.train()
            epoch_time = time.time()
            dataset_num = 0 
            for key, dataset in train_datasets.items():
                dataset_num += 1
                dataset_train_loss= []
                print("-------------{} ({} / {})------------".format(dataset.data_path, dataset_num, len(train_datasets)))
                train_loader = DataLoader(
                    dataset,
                    batch_size = self.args.batch_size,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                    persistent_workers=True,
                    drop_last=False,
                    pin_memory=True)
                
                train_steps = len(train_loader)
                skip_dataset = False  # Flag to skip the dataset if NaN is encountered
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    if skip_dataset:
                        break  # Skip remaining batches of this datase
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :, :], dec_inp], dim=1).float().to(self.device)
                    
                    # convert to one-dim
                    batch_size, seq_len, dims = batch_x.shape
                    batch_x = batch_x.permute(0, 2, 1).contiguous().view(batch_size * dims, seq_len, 1)
                    batch_size, seq_len, dims = batch_y.shape
                    batch_y = batch_y.permute(0, 2, 1).contiguous().view(batch_size * dims, seq_len, 1)
                    
                    # encoder - decoder
                    try:
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                outputs = outputs[0] if self.args.output_attention else outputs
                                outputs = outputs[:, -self.args.pred_len:, :]
                                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                                loss = criterion(outputs, batch_y)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = outputs[0] if self.args.output_attention else outputs
                            outputs = outputs[:, -self.args.pred_len:, :]
                            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                            loss = criterion(outputs, batch_y)

                        # if torch.isnan(loss).any().item():
                        #     print(f"NaN loss detected in dataset {dataset_num} at epoch {epoch + 1}. Skipping this dataset.")
                        #     skip_dataset = True
                        #     break  # Exit the batch loop for this dataset
                        
                        dataset_train_loss.append(loss.item())

                        if (i + 1) % 100 == 0:
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                            speed = (time.time() - time_now) / iter_count
                            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                            iter_count = 0
                            time_now = time.time()

                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()
                    except Exception as e:
                        print(f"Exception occurred: {e}. Skipping dataset {dataset_num}.")
                        skip_dataset = True
                        break  # Exit the batch loop for this dataset

                if not skip_dataset:
                    dataset_train_loss = np.average(dataset_train_loss)
                    total_train_loss.append(dataset_train_loss)
                    print("Dataset training loss:{:.7f}".format(dataset_train_loss))
                else:
                    print(f"Skipped dataset {dataset_num} due to NaN loss.")
               
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            total_train_loss = np.average(total_train_loss)
            
            # 保存检查点
            checkpoint_filename = os.path.join(path, f"checkpoint_epoch_{epoch+1}.pth")
            self._save_checkpoint(epoch, checkpoint_filename)
            
            print(">>>>>>>>>>>>>>Start Validation<<<<<<<<<<<<<<<")
            
            vali_loss = []
            set_num = 0
            for key, dataset in vali_datasets.items():
                set_num += 1
                print("-------------{} ({} / {})------------".format(dataset.data_path, dataset_num, len(vali_datasets)))
                vali_loader = DataLoader(
                    dataset,
                    batch_size = self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    persistent_workers=True,
                    drop_last=False,
                    pin_memory=True)
                loss = self.vali(dataset, vali_loader, criterion)
                print("Vali Set:{} / {} Vali loss:{:.7f}".format(set_num, len(vali_dataset_list), loss))
                vali_loss.append(loss.item())
            vali_loss = np.average(vali_loss)

            print(">>>>>>>>>>>>>>Start Testing<<<<<<<<<<<<<<<")
            test_loss = []
            set_num = 0
            for key, dataset in test_datasets.items():
                set_num += 1
                test_loader = DataLoader(
                    dataset,
                    batch_size = self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    persistent_workers=True,
                    drop_last=False,
                    pin_memory=True)
                loss = self.vali(dataset, test_loader, criterion)
                test_loss.append(loss.item())
                print("Test Set:{} / {} Test loss:{:.7f}".format(set_num, len(test_dataset_list), loss.item()))
            test_loss = np.average(test_loss)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
            #     epoch + 1, train_steps, total_train_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, total_train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            # early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    def test(self, setting, test=0):
        # 确保数据预处理与训练一致
        self.args.data = 'UnivariateDatasetBenchmark'
        test_dataset_list = self.args.test_dataset_list.replace(',', ' ').split()
        test_datasets = {}
        for i in range(len(test_dataset_list)):
            if test_dataset_list[i] in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
                self.args.data_path = 'ETT-small/' + test_dataset_list[i] + '.csv'
            else:
                self.args.data_path = test_dataset_list[i] + '/' + test_dataset_list[i] + '.csv'
            # 使用与训练相同的 pretrain_flag
            test_datasets[f'dataset{i+1}'] = self._get_data(flag='test', pretrain_flag='fine_tuning')
        print(f'Load {len(test_dataset_list)} test datasets successfully.')
        
        # 加载模型时修正多GPU参数键名
        if test:
            print('Loading model...')
            state_dict = torch.load(self.args.model_location, map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # 删除不需要的 key（参考 exp_basic.py 中对 finetune 的处理逻辑）
            keys_to_remove = ['mlp_head', 'pred_dim_change']
            keys_to_remove_set = set(k for k in state_dict if any(r in k for r in keys_to_remove))
            for k in keys_to_remove_set:
                del state_dict[k]
            
            # 如果当前为单卡，去除多GPU训练时的 "module." 前缀
            if self.args.use_multi_gpu:
                load_info = self.model.load_state_dict(state_dict, strict=False)
            else:
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                load_info = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {load_info.missing_keys}")
            print(f"Unexpected keys: {load_info.unexpected_keys}")
        
        # 创建结果保存目录
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        total_loss = []
        mae_total_loss = []
        criterion = nn.MSELoss()  # 与 vali 使用相同的损失函数
        mae_func = nn.L1Loss()
        self.model.eval()          # 显式设置为评估模式
        with torch.no_grad():      # 关闭梯度计算
            set_num = 1
            for key, test_data in test_datasets.items():
                print(f'Testing on No.{set_num} Dataset: {test_data.data_path}')
                test_loader = DataLoader(
                    test_data,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    persistent_workers=True,
                    drop_last=False,
                    pin_memory=True
                )
                dataset_loss = []
                mae_dataset_loss = []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :, :], dec_inp], dim=1).float().to(self.device)
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = outputs[0] if self.args.output_attention else outputs
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[0] if self.args.output_attention else outputs

                    loss = criterion(outputs, batch_y)
                    mae_loss = mae_func(outputs, batch_y)
                    dataset_loss.append(loss.item())
                    mae_dataset_loss.append(mae_loss.item())
                
                avg_loss = np.average(dataset_loss)
                mae_avg_loss = np.average(mae_dataset_loss)
                total_loss.append(avg_loss)
                mae_total_loss.append(mae_avg_loss)
                print(f'Test Set {set_num} / {len(test_dataset_list)} | MSE Loss: {avg_loss:.7f}, MAE Loss: {mae_avg_loss:.7f}')
                set_num += 1
        
        final_loss = np.average(total_loss)
        mae_final_loss = np.average(mae_total_loss)
        print(f'Final Test Loss (MSE): {final_loss:.7f}, Final Test Loss (MAE): {mae_final_loss:.7f}')
        return final_loss
    '''
    def test(self, setting, test=0):
        self.args.data = 'UnivariateDatasetBenchmark'
        test_dataset_list = self.args.test_dataset_list.split()
        test_datasets = {}
        for i in range(len(test_dataset_list)):
            if test_dataset_list[i] in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
                self.args.data_path = 'ETT-small/' + test_dataset_list[i] + '.csv'
            else:
                self.args.data_path = test_dataset_list[i] + '/' + test_dataset_list[i] + '.csv'
            test_datasets[f'dataset{i+1}'] = self._get_data(flag='test', pretrain_flag='fine_tuning')
        print(f'Load {len(test_dataset_list)} test datasets successfully.')
        
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            state_dict = torch.load(self.args.model_location)
            new_state_dict = {}
            for k, v in state_dict['model_state_dict'].items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict, strict=True)
        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        total_mse, total_mae = [], []
        mse_func = nn.MSELoss()
        mae_func = nn.L1Loss()
        self.model.eval()
        with torch.no_grad():
            set_num = 1
            for key, test_data in test_datasets.items():
                preds = []
                trues = []
                print(f'On No.{set_num} Test Dataset:', )
                test_loader = DataLoader(
                    test_data,
                    batch_size = self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    persistent_workers=True,
                    drop_last=False,
                    pin_memory=True)
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # convert to one-dim
                    # batch_size, seq_len, dims = batch_x.shape
                    # batch_x = batch_x.permute(0, 2, 1).contiguous().view(batch_size * dims, seq_len, 1)
                    # batch_size, seq_len, dims = batch_y.shape
                    # batch_y = batch_y.permute(0, 2, 1).contiguous().view(batch_size * dims, seq_len, 1)
                    
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                    # outputs = outputs.detach().cpu().numpy()
                    # batch_y = batch_y.detach().cpu().numpy()
                    
                    mse = mse_func(outputs, batch_y)
                    mae = mae_func(outputs, batch_y)
                    
                    total_mse.append(mse.item())
                    total_mae.append(mae.item())

                    # pred = outputs
                    # true = batch_y

                #     preds.append(pred)
                #     trues.append(true)
                 
                set_num += 1
                
                # preds = np.array(preds)
                # trues = np.array(trues)
                # print('test shape:', preds.shape, trues.shape)
                # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                # print('test shape:', preds.shape, trues.shape)
                # mae, mse, rmse, mape, mspe = metric(preds, trues)
                print('mse:{}, mae:{}'.format(np.average(total_mse), np.average(total_mae)))

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        return
'''
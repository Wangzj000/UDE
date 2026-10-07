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

warnings.filterwarnings('ignore')

from scipy.stats import spearmanr
from scipy.signal import find_peaks

def directional_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    方向准确率：预测的上升/下降趋势与真实是否一致。
    输入: shape = (B, T, D)
    返回: 单个 float
    """
    delta_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    delta_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    correct = (delta_true * delta_pred) > 0
    return np.mean(correct)

def trend_rate_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    趋势变化率误差：预测变化速度与真实变化的差异。
    输入: shape = (B, T, D)
    返回: 单个 float
    """
    delta_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    delta_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    return np.mean(np.abs(delta_true - delta_pred))

def spearman_batch_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman 等级相关性：评估趋势的单调一致性。
    输入: shape = (B, T, D)
    返回: 平均 spearman rho
    """
    B, T, D = y_true.shape
    rho_list = []
    for b in range(B):
        for d in range(D):
            seq_true = y_true[b, :, d]
            seq_pred = y_pred[b, :, d]
            rho, _ = spearmanr(seq_true, seq_pred)
            if not np.isnan(rho):
                rho_list.append(rho)
    return np.mean(rho_list) if rho_list else np.nan

def turning_point_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray, window: int = 2) -> float:
    """
    转折点准确率：预测序列是否准确捕捉趋势转折（峰值/谷值）。
    - window: 允许的偏差步数
    返回: 平均转折点命中率（float）
    """
    B, T, D = y_true.shape
    acc_list = []

    for b in range(B):
        for d in range(D):
            true_seq = y_true[b, :, d]
            pred_seq = y_pred[b, :, d]

            # 找出真实和预测的转折点（峰值 + 谷值）
            true_peaks, _ = find_peaks(true_seq)
            true_valleys, _ = find_peaks(-true_seq)
            true_turns = np.sort(np.concatenate([true_peaks, true_valleys]))

            pred_peaks, _ = find_peaks(pred_seq)
            pred_valleys, _ = find_peaks(-pred_seq)
            pred_turns = np.sort(np.concatenate([pred_peaks, pred_valleys]))

            # 对每个真实转折点，找是否有预测转折点在容差窗口内
            hit_count = 0
            for t in true_turns:
                if np.any(np.abs(pred_turns - t) <= window):
                    hit_count += 1

            if len(true_turns) > 0:
                acc_list.append(hit_count / len(true_turns))

    return np.mean(acc_list) if acc_list else np.nan


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion, multi_index=False):
        total_loss = []
        total_da, total_tre, total_spearman, total_tpa = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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
                        
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
                if multi_index:
                    da = directional_accuracy_np(true.numpy(), pred.numpy())
                    total_da.append(da)
                    tre = trend_rate_error_np(true.numpy(), pred.numpy())
                    total_tre.append(tre)
                    spearman = spearman_batch_np(true.numpy(), pred.numpy())
                    total_spearman.append(spearman)
                    tpa = turning_point_accuracy_np(true.numpy(), pred.numpy(), window=2)
                    total_tpa.append(tpa)
        total_loss = np.average(total_loss)
        self.model.train()
        if multi_index:
            return total_loss, np.average(total_da), np.average(total_tre), np.average(total_spearman), np.average(total_tpa)
        else:
            return total_loss

    def train(self, setting):
        train_data = self._get_data(flag='train')
        train_loader = DataLoader(
        train_data,
        batch_size=self.args.batch_size,
        shuffle=True,
        num_workers=self.args.num_workers,
        drop_last=False)
        
        vali_data = self._get_data(flag='val')
        vali_loader = DataLoader(
        vali_data,
        batch_size=self.args.batch_size,
        shuffle=False,
        num_workers=self.args.num_workers,
        drop_last=False)
        
        test_data = self._get_data(flag='test')
        test_loader = DataLoader(
        test_data,
        batch_size=self.args.batch_size,
        shuffle=False,
        num_workers=self.args.num_workers,
        drop_last=False)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=self.args.min_delta)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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

                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)
            # test_loss, DA, TRE, Spearman, TPA = self.vali(test_loader, criterion, multi_index=True)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # print(f"Directional Accuracy     : {DA:.4f}")
            # print(f"Trend Rate Error         : {TRE:.4f}")
            # print(f"Spearman Correlation     : {Spearman:.4f}")
            # print(f"Turning Point Accuracy   : {TPA:.4f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data = self._get_data(flag='test')
        test_loader = DataLoader(
        test_data,
        batch_size=self.args.batch_size,
        shuffle=False,
        num_workers=self.args.num_workers,
        drop_last=True)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        total_da, total_tre, total_spearman, total_tpa = [], [], [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # if self.args.model == 'iTransformer':
                #     shape = outputs.shape
                #     print(shape)
                #     outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                #     batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y
                
                da = directional_accuracy_np(true, pred)
                total_da.append(da)
                tre = trend_rate_error_np(true, pred)
                total_tre.append(tre)
                spearman = spearman_batch_np(true, pred)
                total_spearman.append(spearman)
                tpa = turning_point_accuracy_np(true, pred, window=2)
                total_tpa.append(tpa)
                
                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print(f"Directional Accuracy     : {np.average(total_da):.4f}")
        print(f"Trend Rate Error         : {np.average(total_tre):.4f}")
        print(f"Spearman Correlation     : {np.average(total_spearman):.4f}")
        print(f"Turning Point Accuracy   : {np.average(total_tpa):.4f}")
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

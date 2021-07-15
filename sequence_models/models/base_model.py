import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchtext.legacy import data
from sklearn.metrics import *
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

SEED = 2021
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseModel(nn.Module):
    def __init__(self, batch_size=64, seed=SEED):
        super(BaseModel, self).__init__()
        torch.manual_seed(seed)

        self.batch_size = batch_size
        self.seed = seed

        self.train_epochs_loss = []
        self.valid_epochs_loss = []
        self.train_epochs_acc = []
        self.valid_epochs_acc = []
        self.train_epochs_f1 = []
        self.valid_epochs_f1 = []

    def fit(self, training_data, epochs=3, do_validation=False, split_ratio=0.2):
        train_data, valid_data = training_data.split(split_ratio=split_ratio,
                                                     random_state=random.seed(SEED))
        train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data),
                                                                    batch_size=self.batch_size,
                                                                    sort_key=lambda x: len(x.text),
                                                                    sort_within_batch=True,
                                                                    device=device)

        # 设置为训练模式
        model = self.train()
        criterion = self.loss_func
        optimizer = self.optim

        best_valid_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            epoch_f1 = 0

            for batch in train_iterator:
                # 在每一个batch后设置0梯度
                optimizer.zero_grad()

                text, text_lengths = batch.text

                # 转换成一维张量
                predictions = model(text, text_lengths).squeeze()

                # 计算损失
                loss = criterion(predictions, batch.label)

                # 计算二分类精度
                acc = self.binary_accuracy(predictions, batch.label)

                # 计算f1 score
                f1 = self.f1_score_(predictions.detach(), batch.label.detach())

                # 反向传播损耗并计算梯度
                loss.backward()

                # 更新权重
                optimizer.step()

                # 损失和精度
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_f1 += f1.item()
            train_loss = epoch_loss / len(train_iterator)
            train_acc = epoch_acc / len(train_iterator)
            train_f1 = epoch_f1 / len(train_iterator)

            self.train_epochs_loss.append(train_loss)
            self.train_epochs_acc.append(train_acc)
            self.train_epochs_f1.append(train_f1)

            print(f'Epoch {epoch + 1}：\n  Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | '
                  f'Train f1: {train_f1 * 100:.2f}%')

            if do_validation:
                valid_loss, valid_acc, valid_f1 = self.evaluate(valid_iterator)

                # 保存最佳模型
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), './outputs/saved_weights.pt')

                self.valid_epochs_loss.append(valid_loss)
                self.valid_epochs_acc.append(valid_acc)
                self.valid_epochs_f1.append(valid_f1)

                print(f'   Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% |  '
                      f'Val. f1: {valid_f1 * 100:.2f}%')

    def evaluate(self, iterator):
        # 预测值
        pred_ans = torch.tensor(self.predict(iterator))
        # 标签
        labels = torch.tensor(np.concatenate([x.label for x in iterator]).astype("float64"))

        loss = self.loss_func(pred_ans, labels)
        accuracy = self.binary_accuracy(pred_ans, labels)
        f1 = self.f1_score_(pred_ans, labels)

        return loss, accuracy, f1

    def predict(self, iterator):
        # 停用dropout层
        model = self.eval()

        # 取消autograd
        pred_ans = []
        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text

                # 转换为一维张量
                predictions = model(text, text_lengths).squeeze()
                pred_ans.append(predictions)

        return np.concatenate(pred_ans).astype("float64")

    def compile(self, optimizer, loss=None, metrics=None):
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def plot_metrics(self, metric='loss', figsize=(6, 4)):
        _ = plt.figure(figsize=figsize)
        x_indices = [i + 1 for i in range(len(self.train_epochs_loss))]
        if metric == 'loss':
            plt.plot(x_indices, self.train_epochs_loss, label='train_' + metric)
            plt.plot(x_indices, self.valid_epochs_loss, label='test_' + metric)
        elif metric == 'acc':
            plt.plot(x_indices, self.train_epochs_acc, label='train_' + metric)
            plt.plot(x_indices, self.valid_epochs_acc, label='test_' + metric)
        elif metric == 'f1':
            plt.plot(x_indices, self.train_epochs_f1, label='train_' + metric)
            plt.plot(x_indices, self.valid_epochs_f1, label='test_' + metric)
        plt.xlabel('epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.show()

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.parameters()))  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()))
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_cross_entropy":
                loss_func = F.binary_cross_entropy
            elif loss == "cross_entropy":
                loss_func = F.cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        pass

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def binary_accuracy(self, preds, y):
        # 四舍五入到最接近的整数
        rounded_preds = torch.round(preds)

        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc

    def f1_score_(self, preds, y):
        rounded_preds = np.round(preds)
        f1 = f1_score(y, rounded_preds)
        return f1

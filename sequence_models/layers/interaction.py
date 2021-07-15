import torch
import torch.nn as nn


def attention_net1(self, x, query, mask=None):
    d_k = query.size(-1)  # d_k为query的维度

    # scores: [batch, seq_len, seq_len] = query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
    # scores: [batch, seq_len, seq_len]
    scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

    # 对最后一个维度 归一化得分
    alpha_n = F.softmax(scores, dim=-1)    # [batch, seq_len, seq_len]
    # 对权重化的x求和
    # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
    context = torch.matmul(alpha_n, x).sum(1)

    return context, alpha_n


def attention_net2(self, lstm_output):
    output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.layer_size])
    attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
    attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
    exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
    alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
    alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
    state = lstm_output.permute(1, 0, 2)
    attn_output = torch.sum(state * alphas_reshape, 1)
    return attn_output

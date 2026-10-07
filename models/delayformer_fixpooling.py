import torch
from torch import nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import TokenEmbedding, DataEmbedding_wo_pos, TemporalEmbedding, TimeFeatureEmbedding, PositionalEmbedding
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, args):
        # if individual: input: [batch_size, seq_len, n_vars] → hankel: [batch_size, n_vars, h=seq_len-n+1, n=w] → 
        # patch: [batch_size*n_var, patch_num=h*w/p1/p2, p1*p2] → projection: [batch_size*n_var, patch_num=h*w/p1/p2, d_model] request: h%p1=w%p2=0
        # if channel-mixed: input: [batch_size, seq_len, n_vars] → hankel: [batch_size, n_vars, h=seq_len-n+1, n=w] → 
        # patch: [batch_size, patch_num=h*w/p1/p2, p1*p2*n_var]
        super(Model, self).__init__()
        self.task_name = args.task_name
        self.individual = args.channel_independence
        self.d_model = args.d_model
        self.pe = args.pe
        self.project = args.project
        self.dropout = args.dropout
        self.n_heads = args.n_heads
        self.e_layers = args.e_layers
        self.d_ff = args.d_ff
        
        # hankelize
        self.h = args.seq_len - args.L + 1
        self.w = args.L
        
        # patching
        self.n_vars = args.n_vars
        self.p1 = args.p1
        self.p2 = args.p2
        self.patch_num = self.h*self.w/self.p1/self.p2 
        if int(self.patch_num) == self.patch_num:
            self.patch_num = int(self.patch_num)
            pass
        else:
            raise RuntimeError('Please change hyper-paramters')
        
        # token pooling
        self.pooling_kernel = args.pooling_kernel
        self.pooling_type = args.pooling_type
        
        self.patch_layer = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.p1, p2 = self.p2), # batch, token_num, token_dim 反向：Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p)
                                        )
        if self.individual:
            self.patch_size = self.p1*self.p2
        else:
            self.patch_size = self.p1*self.p2*self.n_vars 
        
        if self.project == 'linear':
            self.W_projection = nn.Linear(self.patch_size, self.d_model)
        elif self.project == 'conv':
            self.W_projection = TokenEmbedding(self.patch_size, self.d_model)
        
        if self.pe == 'learnable_pe':
            self.position = positional_encoding('zeros', True, self.patch_num, self.d_model)
        elif self.pe == 'fix_pe':
            self.position = PositionalEmbedding(self.d_model)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=self.dropout,
                                      output_attention=False), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation='gelu'
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'pretrain', 'finetune']:
            self.pred_len = args.pred_len
        if self.task_name == 'imputation':
            self.pred_len = args.seq_len
        if self.task_name == 'anomaly_detection':
            self.pred_len = args.seq_len
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2, end_dim=-1))
                self.linears.append(nn.Linear(int(self.patch_num/self.pooling_kernel)*self.d_model, self.pred_len))
                self.dropouts.append(nn.Dropout(self.dropout))
        else:
            self.pred_dim_change = nn.Sequential(nn.Linear(int(self.patch_num/self.pooling_kernel), self.n_vars),
                                                nn.LayerNorm(self.n_vars),
                                                nn.Dropout(self.dropout))
            self.mlp_head = nn.Sequential(nn.Linear(self.d_model, self.pred_len, bias=True),
                                        nn.Dropout(self.dropout))        
        self.revin_layer = RevIN(self.n_vars, affine=True, subtract_last=False)
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
    # def forward(self, x):
        x = self.revin_layer(x, 'norm')
        x = hankel(x, self.w, True)
        if self.individual:
            tokens = [self.patch_layer(x[:,i,:,:].unsqueeze(1)) for i in range(x.shape[1])]
            tokens = torch.stack(tokens, dim=1)
            tokens = torch.reshape(tokens, (tokens.shape[0]*tokens.shape[1], tokens.shape[2], tokens.shape[3])) #(batch_size*n_var, patch_num, d_model)
        else:
            tokens = self.patch_layer(x)
            
        tokens = self.W_projection(tokens)
        if self.pooling_type == 'max':
            enc_in = F.max_pool1d(tokens.permute(0, 2, 1), kernel_size=self.pooling_kernel).permute(0, 2, 1)
        if self.pooling_type == 'avg':
            enc_in = F.avg_pool1d(tokens.permute(0, 2, 1), kernel_size=self.pooling_kernel).permute(0, 2, 1)
            
        if self.pe == 'learnable_pos':
            enc_in = enc_in + self.position
        elif self.pe == 'fix_pos':
            enc_in = enc_in + self.position(enc_in)
        enc_out, _ = self.encoder(enc_in)
        if self.individual:
            enc_out = torch.reshape(enc_out, (-1, self.n_vars, enc_out.shape[-2], enc_out.shape[-1]))   #(batch_size, n_var, patch_num, d_model)
            z = []
            for i in range(self.n_vars):
                out = self.flattens[i](enc_out[:,i,:,:])
                out = self.linears[i](out)
                out = self.dropouts[i](out)
                z.append(out)
            z = torch.stack(z, dim=-1)
        else:
            z = self.pred_dim_change(enc_out.permute(0, 2, 1))
            z = self.mlp_head(z.permute(0, 2, 1)).permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')
        return z
    
    
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

# def one_dim_hankel(x, n):
#     m = len(x) - n + 1
#     return torch.stack([x[i:i+n] for i in range(m)]).squeeze(-1) # [seq_len-n+1, n]

# def hankel(x, n, batch_data=False): # x: Tensor
#     if batch_data: # [batch_size, seq_len, variables]  
#         hankels = []
#         batch_size, _, variables = x.shape
#         for i in range(batch_size):
#             batch = []
#             for j in range(variables):
#                 batch.append(one_dim_hankel(x[i, :, j], n))
#             hankels.append(torch.stack(batch))
#         return torch.stack(hankels) # [batch_size, variables, seq_len-n+1, n]
    
#     else: # [seq_len, variables]
#         hankels = []
#         variables = x.shape[1]
#         for i in range(variables):
#             hankels.append(one_dim_hankel(x[:, i], n))
#         return torch.stack(hankels) # [variables, seq_len-n+1, n]
    
def hankel(x, n, batch_data=False):
    """
    使用矢量化操作优化的hankel函数。
    
    参数:
    - x: 输入张量，形状为[batch_size, seq_len, variables]（如果batch_data=True）
         或者[seq_len, variables]（如果batch_data=False）。
    - n: Hankel矩阵中每个子矩阵的列数。
    - batch_data: 指示输入数据是否包含批次维度的布尔值。
    
    返回:
    - 优化后的Hankel矩阵。
    """
    if batch_data:
        batch_size, seq_len, variables = x.shape
        m = seq_len - n + 1
        idx = torch.arange(n).unsqueeze(0) + torch.arange(m).unsqueeze(1)
        hankels = x[:, idx, :]  # [batch_size, m, n, variables]
        return hankels.permute(0, 3, 1, 2)  # [batch_size, m, n, variables]
    else:
        seq_len, variables = x.shape
        m = seq_len - n + 1
        idx = torch.arange(n).unsqueeze(0) + torch.arange(m).unsqueeze(1)
        hankels = x[idx, :] # [m, n, variables]
        return hankels.permute(2, 0, 1)  # [variables, m, n]
import math
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from einops import rearrange, repeat, einsum
from typing import Dict, List, Optional, Union
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from ..utils import Graph

class DynamicFilterGNN(nn.Module):
    def __init__(self, in_features, out_features, filter_adjacency_matrix, bias=True):
        super(DynamicFilterGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_filter = nn.Parameter(torch.Tensor(in_features, in_features))

        use_gpu = torch.cuda.is_available()
        self.filter_adjacency_matrix = None
        #self.base_filter = nn.Parameter(torch.Tensor(in_features, in_features))
        if use_gpu:
            self.filter_adjacency_matrix = Variable(filter_adjacency_matrix.cuda(), requires_grad=False)
        else:
            self.filter_adjacency_matrix = Variable(filter_adjacency_matrix, requires_grad=False)

        self.transform = nn.Linear(in_features, in_features)
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.base_filter.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        transformed_filter = self.transform(self.base_filter)
        transformed_adjacency = 0.9*self.filter_adjacency_matrix+0.1*transformed_filter
        result_embed = F.linear(input, transformed_adjacency.matmul(self.weight), self.bias)
        #F.linear(input, transformed_adjacency.matmul(self.weight), self.bias)
        return result_embed

    
    def get_transformed_adjacency(self):
        transformed_filter = self.transform(self.base_filter)
        transformed_adjacency = 0.9 * self.filter_adjacency_matrix + 0.1 * transformed_filter
        return transformed_adjacency
    
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'

# KFGN (Kalman Filtering Graph Neural Networks) Model
class KFGN(nn.Module):
    def __init__(self, K, A, feature_size, Clamp_A=True):
        super(KFGN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.K = K
        self.A_list = []

        D_inverse = torch.diag(1 / torch.sum(A, 0))
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A

        A_temp = torch.eye(feature_size, feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, A)
            if Clamp_A:
                A_temp = torch.clamp(A_temp, max=1.)
            self.A_list.append(A_temp)

        self.gc_list = nn.ModuleList([DynamicFilterGNN(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])
        hidden_size = self.feature_size
        gc_input_size = self.feature_size * K

        self.fl = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.il = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(gc_input_size + hidden_size, hidden_size)


        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)


        input_size = self.feature_size

        self.rfl = nn.Linear(input_size + hidden_size, hidden_size)
        self.ril = nn.Linear(input_size + hidden_size, hidden_size)
        self.rol = nn.Linear(input_size + hidden_size, hidden_size)
        self.rCl = nn.Linear(input_size + hidden_size, hidden_size)

        # addtional vars
        self.c = torch.nn.Parameter(torch.Tensor([1]))

        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 64)
        self.fc5 = nn.Linear(64, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, 64)
        
    def forward(self, input, Hidden_State=None, Cell_State=None, rHidden_State=None, rCell_State=None):
        batch_size, time_steps, _ = input.shape
        if Hidden_State is None:
            Hidden_State = Variable(torch.zeros(batch_size,self.feature_size).cuda())
        if Cell_State is None:
            Cell_State = Variable(torch.zeros(batch_size,self.feature_size).cuda())
        if rHidden_State is None:
            rHidden_State = Variable(torch.zeros(batch_size,self.feature_size).cuda())
        if rCell_State is None:
            rCell_State = Variable(torch.zeros(batch_size,self.feature_size).cuda())

        Hidden_State = Hidden_State.unsqueeze(1).expand(-1, time_steps, -1)
        Cell_State = Cell_State.unsqueeze(1).expand(-1, time_steps, -1)
        rHidden_State = rHidden_State.unsqueeze(1).expand(-1, time_steps, -1)
        rCell_State = rCell_State.unsqueeze(1).expand(-1, time_steps, -1)
            
        x = input
        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Hidden_State), 1)
        dim1=combined.shape[0]
        dim2=combined.shape[1]
        dim3=combined.shape[2]
        combined=combined.view(dim1,dim2//4,dim3*4)

        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        # LSTM
        rcombined = torch.cat((input, rHidden_State), 1)
        d1=rcombined.shape[0]
        d2=rcombined.shape[1]
        d3=rcombined.shape[2]
        rcombined=rcombined.view(d1,d2//2,d3*2)
        rf = torch.sigmoid(self.rfl(rcombined))
        ri = torch.sigmoid(self.ril(rcombined))
        ro = torch.sigmoid(self.rol(rcombined))
        rC = torch.tanh(self.rCl(rcombined))
        rCell_State = rf * rCell_State + ri * rC
        rHidden_State = ro * torch.tanh(rCell_State)

        # Kalman Filtering
        var1, var2 = torch.var(input), torch.var(gc)

        pred = (Hidden_State * var1 * self.c + rHidden_State * var2) / (var1 + var2 * self.c)

        return pred
        #return Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State, rHidden_State, rCell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred = self.forward(
                torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State, rHidden_State, rCell_State)
        return pred


    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            rCell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State, rHidden_State, rCell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            rCell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State, rHidden_State, rCell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            rHidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            rCell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State, rHidden_State, rCell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            rHidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            rCell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State, rHidden_State, rCell_State


@MODELS.register_module()
class KFGN_Mamba(BaseModule):
    def __init__(self,
                graph_cfg: Dict,
                features: int,
                d_model: int = 64,
                n_layer: int = 4,
                d_state: int = 16,
                expand: int = 2,
                dt_rank: Union[int, str] = 'auto',
                d_conv: int = 4,
                conv_bias: bool = True,
                bias: bool = False,
                K: int = 3,

                #  graph_cfg: Dict,
                #  in_channels: int = 3,
                #  base_channels: int = 64,
                #  data_bn_type: str = 'VC',
                #  ch_ratio: int = 2,
                #  num_person: int = 2,
                #  num_stages: int = 10,
                #  inflate_stages: List[int] = [5, 8],
                #  down_stages: List[int] = [5, 8],
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        feature_size=A.shape[0]

        d_inner = int(expand * d_model)
        if dt_rank == 'auto':
            dt_rank = math.ceil(d_model / 16)

        self.kfgn = KFGN(K=K, A=A, feature_size=feature_size)
        self.encode = nn.Linear(features, d_model)
        self.encoder_layers = nn.ModuleList([ResidualBlock(K, A, feature_size, d_model, d_inner, bias, conv_bias, d_conv, dt_rank, d_state, self.kfgn) for _ in range(n_layer)])
        self.encoder_norm = RMSNorm(d_model)
        # Decoder (identical to Encoder)
        ##self.decoder_layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)]) #You can optionally uncommand these lines to use the identical Decoder.
        ##self.decoder_norm = RMSNorm(args.d_model) #You can optionally uncommand these lines to use the identical Decoder.
        self.decode = nn.Linear(d_model, features)

    def forward(self, input_ids):
        x = self.encode(input_ids)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)
        # Decoder
        ##for layer in self.decoder_layers:#You can optionally uncommand these lines to use the identical Decoder.
        ##    x = layer(x) #You can optionally uncommand these lines to use the identical Decoder.
        ##x = self.decoder_norm(x) #You can optionally uncommand these lines to use the identical Decoder.
        
        # Output
        x = self.decode(x)
        
        return x


# Residual Block in Mamba Model
class ResidualBlock(nn.Module):
    def __init__(self, K, A, feature_size, d_model, d_inner, bias, conv_bias, d_conv, dt_rank, d_state, kfgn: KFGN):
        super().__init__()
        self.kfgn = KFGN(K=K, A=A, feature_size=feature_size)       
        self.mixer = MambaBlock(d_model, d_inner, bias, conv_bias, d_conv, dt_rank, d_state, kfgn)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x0 = x
        x1 = self.norm(x)
        x2 = self.kfgn(x1)
        x3 = self.mixer(x2)
        output = x3 + x1
        
        return output


class MambaBlock(nn.Module):    
    def __init__(self, d_model, d_inner, bias, conv_bias, d_conv, dt_rank, d_state, kfgn: KFGN):
        super().__init__()
        self.kfgn = kfgn

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        

    def forward(self, x):
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
    
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        # This is the new version of Selective Scan Algorithm named as "Graph Selective Scan"
        #In Graph Selective Scan, we use the Feed-Forward graph information from KFGN, and incorporate the Feed-Forward information with "delta"
        temp_adj = self.kfgn.gc_list[-1].get_transformed_adjacency()        
        temp_adj_padded = torch.ones(d_in, d_in, device=temp_adj.device)       
        temp_adj_padded[:temp_adj.size(0), :temp_adj.size(1)] = temp_adj
        
        delta_p = torch.matmul(delta,temp_adj_padded)
        
        # The fused param delta_p will participate in the following upgrading of deltaA and deltaB_u
        deltaA = torch.exp(einsum(delta_p, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta_p, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y

    
class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
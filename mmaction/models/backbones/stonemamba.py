import math
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Dict, List, Optional, Union
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from ..utils import Graph, unit_gcn, mstcn, unit_tcn

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

@dataclass
class ModelArgs:
    d_model: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 3
    conv_bias: bool = True
    bias: bool = False
    
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

            

@MODELS.register_module()
class StoneMamba(BaseModule):
    def __init__(self,
                 graph_cfg: Dict,
                 in_channels: int = 3,
                 num_person: int = 2,
                 d_model_base: int = 16,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        

        # Graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        num_joint = A.size(-1)
        A_outward = self.graph.A_outward_binary
        I = np.eye(self.graph.num_node)
        self.A_vector = torch.from_numpy(I - np.linalg.matrix_power(A_outward, 8))     
        self.p = torch.tensor(self.A_vector,dtype=torch.float)
        # Data Normalization
        self.data_bn = nn.BatchNorm1d(num_person * d_model_base * num_joint)
        nn.init.constant_(self.data_bn.weight, 1)
        nn.init.constant_(self.data_bn.bias, 0)

        # Embedding        
        self.to_joint_embedding = nn.Linear(in_channels, d_model_base)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joint, d_model_base))

        self.l1 = ResidualBlock(d_model_base, d_model_base, A, residual=False)
        self.l2 = ResidualBlock(d_model_base, d_model_base, A)
        self.l3 = ResidualBlock(d_model_base, d_model_base, A)
        self.l4 = ResidualBlock(d_model_base, d_model_base, A)
        self.l5 = ResidualBlock(d_model_base, d_model_base*2, A, stride=2)
        self.l6 = ResidualBlock(d_model_base*2, d_model_base*2, A)
        self.l7 = ResidualBlock(d_model_base*2, d_model_base*2, A)
        self.l8 = ResidualBlock(d_model_base*2, d_model_base*3, A, stride=2)
        self.l9 = ResidualBlock(d_model_base*3, d_model_base*3, A)
        self.l10 = ResidualBlock(d_model_base*3, d_model_base*3, A)

        self.first_tram = nn.Sequential(
                nn.AvgPool2d((4,1)),
                nn.Conv2d(d_model_base, d_model_base*3, 1),
                nn.BatchNorm2d(d_model_base*3),
                nn.ReLU()
            )
        self.second_tram = nn.Sequential(
                nn.AvgPool2d((2,1)),
                nn.Conv2d(d_model_base*2, d_model_base*3, 1),
                nn.BatchNorm2d(d_model_base*3),
                nn.ReLU()
            )

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = rearrange(x, 'n m t v c -> (n m t) v c').contiguous()


        
        
        
        x = self.p.to(x.device).expand(N*M*T, -1, -1) @ x   

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :V]

        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x2 = x
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x3 = x
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        
        x2 = self.first_tram(x2)
        x3 = self.second_tram(x3)
        x = x + x2 + x3

        x = rearrange(x, '(n m) c t v -> n m (v c) t', m=M).contiguous()

        x = x.view(x.shape + (1,))
        
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        
        self.gcn = Shift_gcn(in_channels, out_channels, A)
        self.tcn = unit_tcn(out_channels, out_channels, stride=stride)
        self.mixer = MambaBlock(ModelArgs(d_model=out_channels*A.size(-1)))
        self.norm = RMSNorm(out_channels*A.size(-1))
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        
            
        

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, T, V)

        Returns:
            Tensor of shape (N, C, T, V)
        """

        res = self.residual(x)
        x = self.tcn(self.gcn(x))
        N, C, T, V = x.size()
        x = rearrange(x, 'n c t v -> n t (v c)')
        x = self.mixer(self.norm(x))
        x = rearrange(x, 'n t (v c) -> n c t v', v=V, c=C)
        x = x + res
        return self.relu(x)


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        num_joints = A.size(-1)
        
        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True),requires_grad=True)
        nn.init.constant_(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,num_joints,in_channels, requires_grad=True),requires_grad=True)
        nn.init.constant_(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(num_joints*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        index_array = np.empty(num_joints*in_channels).astype(np.int32)
        for i in range(num_joints):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*25)
        self.shift_in = torch.tensor(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(num_joints*out_channels).astype(np.int32)
        for i in range(num_joints):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*25)
        self.shift_out = torch.tensor(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):

        self.shift_in = self.shift_in.to(x0.device)
        self.shift_out = self.shift_out.to(x0.device)
        self.Feature_Mask = self.Feature_Mask.to(x0.device)
        self.Linear_weight = self.Linear_weight.to(x0.device)
        self.Linear_bias = self.Linear_bias.to(x0.device)

        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)
        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x
        
class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner, bias=args.bias)

        self.conv1d_x = nn.Conv1d(
            in_channels=args.d_inner//2,
            out_channels=args.d_inner//2,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner//2,
            padding=1
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=args.d_inner//2,
            out_channels=args.d_inner//2,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner//2,
            padding=1
        )
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner//2, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner//2, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner//2)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(args.d_inner//2))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        xz = self.in_proj(x)  # shape (b, l, 2 * d_in)
        xz = rearrange(xz, 'b l d_in -> b d_in l')
        x, z = xz.chunk(2, dim=1)

        # x: shape (b, d_in, l)

        x = F.silu(self.conv1d_x(x))
        x = rearrange(x, 'b d_in l -> b l d_in')

        z = F.silu(self.conv1d_z(z))
        z = rearrange(z, 'b d_in l -> b l d_in')
        
        y = self.ssm(x)
        y = torch.cat([y, z], dim=2)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
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


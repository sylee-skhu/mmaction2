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
from mmcv.cnn import ConvModule
from ..utils import Graph, unit_gcn, mstcn, unit_tcn
from timm.models.layers import Mlp, DropPath, trunc_normal_

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
                 d_model_base: int = 64,
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
        trunc_normal_(self.pos_embedding, std=.02)

        self.l1 = ResidualBlock(d_model_base, A)
        self.l2 = ResidualBlock(d_model_base, A)
        self.l3 = ResidualBlock(d_model_base, A)
        self.l4 = ResidualBlock(d_model_base, A)
        self.l5 = ResidualBlock(d_model_base, A)
        self.l6 = ResidualBlock(d_model_base, A)
        self.l7 = ResidualBlock(d_model_base, A)
        self.l8 = ResidualBlock(d_model_base, A)
        self.l9 = ResidualBlock(d_model_base, A)

        self.first_tram = nn.Sequential(
                nn.Conv2d(d_model_base, d_model_base, 1),
                nn.BatchNorm2d(d_model_base),
                nn.ReLU()
            )
        self.second_tram = nn.Sequential(
                nn.Conv2d(d_model_base, d_model_base, 1),
                nn.BatchNorm2d(d_model_base),
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
        x2 = x
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x3 = x
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        
        x2 = self.first_tram(x2)
        x3 = self.second_tram(x3)
        x = x + x2 + x3

        x = rearrange(x, '(n m) c t v -> n m (v c) t', m=M).contiguous()

        x = x.view(x.shape + (1,))
        
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, A, drop=0.1, drop_path=0.1):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()

        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_features=in_channels, out_features=2 * in_channels, bias=True)
        
        self.compute_A1 = ConvModule(in_channels // 2, in_channels // 2, kernel_size=1, bias=True)
        self.compute_A2 = ConvModule(in_channels // 2, in_channels // 2, kernel_size=1, bias=True)

        self.tconv = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=(9, 1),
                               padding=((9 - 1) // 2, 0), groups=8)

        self.mixer = MambaBlock(ModelArgs(d_model=in_channels*A.size(-1)))
        self.norm = RMSNorm(in_channels*A.size(-1))

        self.out_proj = nn.Linear(in_features=in_channels*2, out_features=in_channels, bias=True)
        self.out_norm = nn.LayerNorm(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=int(4 * in_channels),
                       act_layer=nn.GELU, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            
        

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, T, V)

        Returns:
            Tensor of shape (N, C, T, V)
        """

        N, C, T, V = x.shape

        x = x.permute(0, 2, 3, 1).contiguous() # N, T, V, C
        skip = x

        x = self.in_proj(self.in_norm(x)).permute(0, 3, 1, 2).contiguous() # N, 2C, T, V

        x_spatial, x_temporal, x_mamba = torch.split(x, [C // 2, C // 2, C], dim=1) # x_spatial: N, C/2, T, V | x_temporal: N, C/2, T, V | x_mamba: N, C, T, V

        out = []

        # Spatial
        A1 = self.compute_A1(x_spatial).permute(0, 2, 3, 1).contiguous()  # A1: N, T, V, C/2
        A2 = self.compute_A2(x_spatial).permute(0, 2, 1, 3).contiguous()  # A2: N, T, C/2, V
        A = A1.matmul(A2) # A: N, T, V, V
        A = nn.Softmax(dim=-1)(A)
        out_s = torch.einsum('n c t v, n t v v -> n c t v', x_spatial, A) # z: N, C/2, T, V
        out.append(out_s)

        # Temporal
        out_t = self.tconv(x_temporal) # N, C/2, T, V
        out.append(out_t)

        # Mamba
        out_mamba = rearrange(x_mamba, 'n c t v -> n t (v c)')
        out_mamba = self.mixer(self.norm(out_mamba))+out_mamba
        out_mamba = rearrange(out_mamba, 'n t (v c) -> n c t v', v=V, c=C)
        out.append(out_mamba)

        output = self.out_proj(torch.cat(out, dim=1).permute(0, 2, 3, 1).contiguous())
        output = skip + self.drop_path(output)

        output = output + self.drop_path(self.mlp(self.out_norm(output)))
        output = output.permute(0, 3, 1, 2).contiguous()
        
        return output

        
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


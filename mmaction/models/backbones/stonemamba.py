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

from mamba_ssm import Mamba, Mamba2

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

        self.mamba = Mamba(d_model=in_channels*A.size(-1), d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(in_channels*A.size(-1))

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
        out_mamba = self.norm(self.mamba(out_mamba)+out_mamba)
        out_mamba = rearrange(out_mamba, 'n t (v c) -> n c t v', v=V, c=C)
        out.append(out_mamba)

        output = self.out_proj(torch.cat(out, dim=1).permute(0, 2, 3, 1).contiguous())
        output = skip + self.drop_path(output)

        output = output + self.drop_path(self.mlp(self.out_norm(output)))
        output = output.permute(0, 3, 1, 2).contiguous()
        
        return output

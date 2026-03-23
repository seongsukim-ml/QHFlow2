import math
import collections
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_scatter import scatter
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_max_pool

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Linear, TensorProduct, Irreps
from e3nn.o3._norm import Norm
from e3nn.math import normalize2mom, perm
from e3nn.util.jit import compile_mode
from .layers import *
from .modules import *


def construct_o3irrps(dim,order):
    string = []
    for l in range(order+1):
        string.append(f"{dim}x{l}e" if l%2==0 else f"{dim}x{l}o")
    return "+".join(string)

def construct_o3irrps_base(dim,order):
    string = []
    for l in range(order+1):
        string.append(f"{dim}x{l}e")
    return "+".join(string)

def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    """Generate sinusoidal time embeddings for diffusion timesteps.
    
    Args:
        timesteps: Timestep values
        embedding_dim: Dimension of output embedding
        max_positions: Maximum number of positions for encoding
        
    Returns:
        torch.Tensor: Time embeddings with sinusoidal encoding
    """
    # Code adapted from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class PureExpansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super(PureExpansion, self).__init__()
        self.irrep_in = Irreps(irrep_in)
        self.irrep_out_1 = Irreps(irrep_out_1)
        self.irrep_out_2 = Irreps(irrep_out_2)
        self.instructions = self.get_expansion_path(self.irrep_in, self.irrep_out_1, self.irrep_out_2)

    def PureContraction(self):
        return PureContraction(self.irrep_out_1, self.irrep_out_2, self.irrep_in)

    def forward(self, x_in):
        batch_num = x_in.shape[0]
        x_in_splited = [
            x_in[:, s].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim) 
            for s, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)
        ]
    
        outputs = {}
        for ins in self.instructions:
            idx_in, idx_out1, idx_out2 = ins[0], ins[1], ins[2]
            l_in, l_out1, l_out2 = ins[3], ins[4], ins[5]
            mul_tuple = ins[-1]

            mul_ir_out1 = self.irrep_out_1[idx_out1]
            mul_ir_out2 = self.irrep_out_2[idx_out2]

            x1 = x_in_splited[idx_in]
            w3j_matrix = o3.wigner_3j(l_out1, l_out2, l_in).to(x_in.device).type(x1.dtype)
            
            mul_channel = torch.ones(mul_tuple, device=x_in.device, dtype=x1.dtype)
            
            result = torch.einsum(
                "wuv, ijk, bwk -> buivj",
                mul_channel,
                w3j_matrix,
                x1,
            )
            
            norm_factor = (2 * l_in + 1) ** 0.5
            result = result * norm_factor

            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            
            key = (idx_out1, idx_out2)
            if key in outputs:
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result

        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                key = (i, j)
                if key not in outputs:
                    blocks.append(
                        torch.zeros(
                            (x_in.shape[0], self.irrep_out_1[i].dim, self.irrep_out_2[j].dim),
                            device=x_in.device,
                            dtype=x_in.dtype
                        )
                    )
                else:
                    blocks.append(outputs[key])
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output

    @staticmethod
    def get_expansion_path(irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        path_info = [i, j, k, ir_in.l, ir_out1.l, ir_out2.l, [num_in, num_out1, num_out2]]
                        instructions.append(path_info)
        return instructions
    
class PureContraction(nn.Module):
    def __init__(self, irrep_in_1, irrep_in_2, irrep_out):
        super(PureContraction, self).__init__()
        self.irrep_in_1 = Irreps(irrep_in_1)
        self.irrep_in_2 = Irreps(irrep_in_2)
        self.irrep_out = Irreps(irrep_out)
        self.instructions = self.get_contraction_path(self.irrep_in_1, self.irrep_in_2, self.irrep_out)
        
        self.path_counts = self._count_paths()

    def PureExpansion(self):
        return PureExpansion(self.irrep_in_1, self.irrep_in_2, self.irrep_out)

    def _count_paths(self):
        """각 출력 irrep 인덱스(idx_out)에 기여하는 경로의 수를 계산합니다."""
        counts = torch.zeros(len(self.irrep_out), dtype=torch.float32)
        for ins in self.instructions:
            idx_out = ins[2]
            counts[idx_out] += 1 * ins[-1][0] * ins[-1][1]
        return counts.clamp(min=1)

    def forward(self, x_in):
        batch_num = x_in.shape[0]
        x_in_blocks = [
            [x_in[:, s1, s2] for s2 in self.irrep_in_2.slices()]
            for s1 in self.irrep_in_1.slices()
        ]

        outputs = {}
        for ins in self.instructions:
            idx_in1, idx_in2, idx_out = ins[0], ins[1], ins[2]
            l_in1, l_in2, l_out = ins[3], ins[4], ins[5]
            mul_tuple = [ins[-1][2], ins[-1][0], ins[-1][1]]

            mul_ir_in1 = self.irrep_in_1[idx_in1]
            mul_ir_in2 = self.irrep_in_2[idx_in2]

            x1_reshaped = x_in_blocks[idx_in1][idx_in2].reshape(
                batch_num, mul_ir_in1.mul, mul_ir_in1.ir.dim, mul_ir_in2.mul, mul_ir_in2.ir.dim
            )
            
            w3j_matrix = o3.wigner_3j(l_in1, l_in2, l_out).to(x_in.device).type(x1_reshaped.dtype)
            mul_channel = torch.ones(mul_tuple, device=x_in.device, dtype=x1_reshaped.dtype)

            result = torch.einsum("buivj, ijk, wuv -> bwk", x1_reshaped, w3j_matrix, mul_channel)
            norm_factor = (2 * l_out + 1) ** 0.5
            result = result * norm_factor

            key = idx_out
            if key in outputs:
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result
    
        final_tensors = []
        path_counts = self.path_counts.to(x_in.device) 

        for i, mul_ir in enumerate(self.irrep_out):
            if i in outputs:
                normalized_output = outputs[i] / path_counts[i]
                final_tensors.append(normalized_output.reshape(batch_num, -1))
            else:
                final_tensors.append(
                    torch.zeros(batch_num, mul_ir.dim, device=x_in.device, dtype=x_in.dtype)
                )
        output = torch.cat(final_tensors, dim=-1)
        return output
    
    @staticmethod
    def get_contraction_path(irrep_in_1, irrep_in_2, irrep_out):
        instructions = []
        for i, (num_in1, ir_in1) in enumerate(irrep_in_1):
            for j, (num_in2, ir_in2) in enumerate(irrep_in_2):
                for k, (num_out, ir_out) in enumerate(irrep_out):
                    if ir_out in ir_in1 * ir_in2:
                        path_info = [i, j, k, ir_in1.l, ir_in2.l, ir_out.l, [num_in1, num_in2, num_out]]
                        instructions.append(path_info)
        return instructions
    
class ParamContraction(nn.Module):
    """
    ParamContraction is a contraction layer that has parameters for each path.
    """
    def __init__(self, irrep_in_1, irrep_in_2, irrep_out, use_bias=True):
        super(ParamContraction, self).__init__()
        self.irrep_in_1 = Irreps(irrep_in_1)
        self.irrep_in_2 = Irreps(irrep_in_2)
        self.irrep_out = Irreps(irrep_out)
        self.instructions = self.get_contraction_path(self.irrep_in_1, self.irrep_in_2, self.irrep_out)

        self.use_bias = use_bias

        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions)
        self.num_bias = sum(prod(ins[-1][2:]) for ins in self.instructions)

        self.path_counts = self._count_paths()

        if self.num_path_weight > 0:
            self.path_weights = nn.Parameter(torch.rand(self.num_path_weight))
        if self.num_bias > 0 and self.use_bias:
            self.bias_weights = nn.Parameter(torch.rand(self.num_bias))
        self.num_weights = self.num_path_weight + self.num_bias

    def _count_paths(self):
        """각 출력 irrep 인덱스(idx_out)에 기여하는 경로의 수를 계산합니다."""
        counts = torch.zeros(len(self.irrep_out), dtype=torch.float32)
        for ins in self.instructions:
            idx_out = ins[2]
            counts[idx_out] += 1 * ins[-1][0] * ins[-1][1]
        return counts.clamp(min=1)

    def forward(self, x_in, weights=None, bias_weights=None):
        batch_num = x_in.shape[0]
        x_in_blocks = [
            [x_in[:, s1, s2] for s2 in self.irrep_in_2.slices()]
            for s1 in self.irrep_in_1.slices()
        ]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            idx_in1, idx_in2, idx_out = ins[0], ins[1], ins[2]
            l_in1, l_in2, l_out = ins[3], ins[4], ins[5]
            mul_tuple = [ins[-1][2], ins[-1][0], ins[-1][1]]

            mul_ir_in1 = self.irrep_in_1[idx_in1]
            mul_ir_in2 = self.irrep_in_2[idx_in2]

            x1_reshaped = x_in_blocks[idx_in1][idx_in2].reshape(
                batch_num, mul_ir_in1.mul, mul_ir_in1.ir.dim, mul_ir_in2.mul, mul_ir_in2.ir.dim
            )
            w3j_matrix = o3.wigner_3j(l_in1, l_in2, l_out).to(x_in.device).type(x1_reshaped.dtype)
            # mul_channel = torch.ones(mul_tuple, device=x_in.device, dtype=x1_reshaped.dtype)

            if weights is None:
                weight = self.path_weights[
                    flat_weight_index : flat_weight_index + prod(ins[-1])
                ].reshape(mul_tuple)
                result = torch.einsum("buivj, ijk, wuv -> bwk", x1_reshaped, w3j_matrix, weight)
                if self.use_bias:
                    bias_weight = self.bias_weights[
                        bias_weight_index : bias_weight_index + prod(ins[-1][2:])
                    ].reshape(ins[-1][2:])
                    bias_weight_index += prod(ins[-1][2:])
                    result = result + bias_weight.unsqueeze(-1)
            else:
                weight = weights[
                    :, flat_weight_index : flat_weight_index + prod(ins[-1])
                ].reshape([-1] + mul_tuple)
                result = torch.einsum("buivj, ijk, wuv -> bwk", x1_reshaped, w3j_matrix, weight)
                if self.use_bias and bias_weights is not None:
                    bias_weight = bias_weights[
                        :, bias_weight_index : bias_weight_index + prod(ins[-1][2:])
                    ].reshape([-1] + ins[-1][2:])
                    bias_weight_index += prod(ins[-1][2:])
                    result = result + bias_weight.unsqueeze(-1)

            flat_weight_index += prod(ins[-1])
            # result = torch.einsum("buivj, ijk, wuv -> bwk", x1_reshaped, w3j_matrix, mul_channel)

            norm_factor = (2 * l_out + 1) ** 0.5
            result = result * norm_factor

            key = idx_out
            if key in outputs:
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result
    
        final_tensors = []
        path_counts = self.path_counts.to(x_in.device) 

        for i, mul_ir in enumerate(self.irrep_out):
            if i in outputs:
                normalized_output = outputs[i] / path_counts[i]
                final_tensors.append(normalized_output.reshape(batch_num, -1))
            else:
                final_tensors.append(
                    torch.zeros(batch_num, mul_ir.dim, device=x_in.device, dtype=x_in.dtype)
                )
        output = torch.cat(final_tensors, dim=-1)
        return output
    
    @staticmethod
    def get_contraction_path(irrep_in_1, irrep_in_2, irrep_out):
        instructions = []
        for i, (num_in1, ir_in1) in enumerate(irrep_in_1):
            for j, (num_in2, ir_in2) in enumerate(irrep_in_2):
                for k, (num_out, ir_out) in enumerate(irrep_out):
                    if ir_out in ir_in1 * ir_in2:
                        path_info = [i, j, k, ir_in1.l, ir_in2.l, ir_out.l, [num_in1, num_in2, num_out]]
                        instructions.append(path_info)
        return instructions
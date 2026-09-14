import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from typing import Dict


class CortexBlock(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.d_model = config.hidden_size
        self.d_ffn = config.intermediate_size
        cortex_config = config.cortex_config
        self.num_bases = cortex_config.get('num_bases', 32)
        self.lora_rank = cortex_config.get('lora_rank', 8)
        self.max_iterations = cortex_config.get('max_iterations', 2)
        self.gate_branch = cortex_config.get('gate_branch', False)
        self.activation = nn.SiLU()
        control_tensor_dim = (
            self.d_model + self.num_bases * (2 if self.gate_branch else 1) +
            (self.d_ffn * self.lora_rank) +
            (self.lora_rank * self.d_ffn) + self.d_ffn + self.d_ffn +
            (self.max_iterations + 1)
        )
        self.context_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2), self.activation,
            nn.Linear(self.d_model * 2, control_tensor_dim)
        )
        self.basis_projection_weights = nn.Parameter(
            torch.randn(self.num_bases, self.d_model, self.d_ffn) * (1 / (self.d_model ** 0.5))
        )
        self.refinement_ffn = nn.Sequential(
            nn.Linear(self.d_ffn, self.d_ffn), self.activation,
            nn.Linear(self.d_ffn, self.d_ffn)
        )
        self.final_down_projection = nn.Linear(self.d_ffn, self.d_model)
        self.record_iterations = False
        self.last_iteration_counts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.basis_projection_weights.dtype)
        batch_size, seq_len, _ = x.shape
        control_signals = self.context_head(x)
        split_dims = [
            self.d_model, self.num_bases * (2 if self.gate_branch else 1),
            self.d_ffn * self.lora_rank, self.lora_rank * self.d_ffn,
            self.d_ffn, self.d_ffn, self.max_iterations + 1
        ]
        (input_gate, basis_weights_raw, lora_A_flat, lora_B_flat,
         neuron_scaler, generated_bias, iteration_logits) = torch.split(control_signals, split_dims, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        x_gated = x * input_gate
        if self.gate_branch:
            basis_up_raw, basis_gate_raw = torch.split(basis_weights_raw, [self.num_bases, self.num_bases], dim=-1)
        else:
            basis_up_raw, basis_gate_raw = basis_weights_raw, None
        basis_weights = F.softmax(basis_up_raw, dim=-1)
        W_up = torch.einsum('bsn,nio->bsio', basis_weights, self.basis_projection_weights)
        hidden = torch.einsum('bsi,bsio->bso', x_gated, W_up)
        lora_A = lora_A_flat.view(batch_size, seq_len, self.d_ffn, self.lora_rank)
        lora_B = lora_B_flat.view(batch_size, seq_len, self.lora_rank, self.d_ffn)
        lora_correction = torch.einsum('bso,bsor->bsr', hidden, lora_A)
        lora_correction = torch.einsum('bsr,bsro->bso', lora_correction, lora_B)
        pre_activation = hidden + lora_correction + generated_bias
        if self.gate_branch:
            gate_weights = F.softmax(basis_gate_raw, dim=-1)
            W_gate = torch.einsum('bsn,nio->bsio', gate_weights, self.basis_projection_weights)
            gate_pre = torch.einsum('bsi,bsio->bso', x, W_gate)
            activated = self.activation(gate_pre) * pre_activation * neuron_scaler
        else:
            activated = self.activation(pre_activation) * neuron_scaler
        iteration_count = torch.argmax(iteration_logits, dim=-1)
        if self.record_iterations:
            self.last_iteration_counts = iteration_count.detach()
        final_state = activated
        if self.max_iterations > 0:
            for i in range(self.max_iterations):
                mask = (iteration_count > i).to(final_state.dtype).unsqueeze(-1)
                refined_chunk = self.refinement_ffn(final_state)
                final_state = final_state + refined_chunk * mask
        output = self.final_down_projection(final_state)
        return output


class CortexBranchMLP(nn.Module):
    def __init__(self, config: Dict, orig_mlp: nn.Module):
        super().__init__()
        self.gate_proj = orig_mlp.gate_proj
        self.up_proj = orig_mlp.up_proj
        self.down_proj = orig_mlp.down_proj
        self.act_fn = orig_mlp.act_fn
        self.branch = _new_block(config, orig_mlp)
        zero_branch_output(self.branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return base + self.branch(x)


def zero_branch_output(block: CortexBlock):
    with torch.no_grad():
        block.final_down_projection.weight.zero_()
        block.final_down_projection.bias.zero_()


def warm_start_block(block: CortexBlock, mlp: nn.Module):
    up = mlp.up_proj.weight.detach().float()
    down = mlp.down_proj.weight.detach().float()
    gate = mlp.gate_proj.weight.detach().float() if block.gate_branch else None
    scaler_offset = block.d_model + block.num_bases * (2 if block.gate_branch else 1) + \
        2 * block.d_ffn * block.lora_rank
    with torch.no_grad():
        block.context_head[0].weight.zero_()
        block.context_head[0].bias.zero_()
        block.context_head[2].weight.zero_()
        bias = block.context_head[2].bias
        bias.zero_()
        bias[:block.d_model] = 8.0
        bias[block.d_model] = 20.0
        if block.gate_branch:
            bias[block.d_model + block.num_bases + 1] = 20.0
        bias[scaler_offset:scaler_offset + block.d_ffn] = 1.0
        block.basis_projection_weights.zero_()
        block.basis_projection_weights[0] = up.T.contiguous()
        if block.gate_branch:
            block.basis_projection_weights[1] = gate.T.contiguous()
        block.refinement_ffn[0].weight.zero_()
        block.refinement_ffn[0].bias.zero_()
        block.refinement_ffn[2].weight.zero_()
        block.refinement_ffn[2].bias.zero_()
        block.final_down_projection.weight.copy_(down)
        block.final_down_projection.bias.zero_()


def _is_ffn(layer: nn.Module) -> bool:
    return hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj')


def _new_block(config, like: nn.Module) -> CortexBlock:
    ref = next(like.parameters())
    return CortexBlock(config).to(device=ref.device, dtype=ref.dtype)


def _set_cortex_config(model, num_bases, lora_rank, max_iterations, gate_branch=False):
    model.config.cortex_config = {
        'num_bases': num_bases,
        'lora_rank': lora_rank,
        'max_iterations': max_iterations,
        'gate_branch': gate_branch,
    }
    return model.config


def replace_ffn_with_cortex(model, every_n_layers: int = 3, num_bases: int = 16,
                            lora_rank: int = 8, max_iterations: int = 2,
                            warm_start: bool = False, gate_branch: bool = False):
    config = _set_cortex_config(model, num_bases, lora_rank, max_iterations, gate_branch)
    replaced = []
    for idx, layer in enumerate(model.model.layers):
        if not _is_ffn(layer) or idx % every_n_layers != 0:
            continue
        block = _new_block(config, layer.mlp)
        if warm_start:
            warm_start_block(block, layer.mlp)
        layer.mlp = block
        replaced.append(idx)
    return replaced


def add_cortex_branches(model, every_n_layers: int = 3, num_bases: int = 16,
                        lora_rank: int = 4, max_iterations: int = 2,
                        gate_branch: bool = False):
    config = _set_cortex_config(model, num_bases, lora_rank, max_iterations, gate_branch)
    added = []
    for idx, layer in enumerate(model.model.layers):
        if not _is_ffn(layer) or idx % every_n_layers != 0:
            continue
        layer.mlp = CortexBranchMLP(config, layer.mlp)
        added.append(idx)
    return added


class CortexForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        cortex_cfg = getattr(config, 'cortex_config', {}) or {}
        replace_ffn_with_cortex(
            self,
            every_n_layers=cortex_cfg.get('every_n_layers', 3),
            num_bases=cortex_cfg.get('num_bases', 16),
            lora_rank=cortex_cfg.get('lora_rank', 4),
            max_iterations=cortex_cfg.get('max_iterations', 2),
            warm_start=False,
            gate_branch=cortex_cfg.get('gate_branch', False),
        )


class CortexParallelForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        cortex_cfg = getattr(config, 'cortex_config', {}) or {}
        add_cortex_branches(
            self,
            every_n_layers=cortex_cfg.get('every_n_layers', 3),
            num_bases=cortex_cfg.get('num_bases', 16),
            lora_rank=cortex_cfg.get('lora_rank', 4),
            max_iterations=cortex_cfg.get('max_iterations', 2),
            gate_branch=cortex_cfg.get('gate_branch', False),
        )

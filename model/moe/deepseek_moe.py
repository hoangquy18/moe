# coding=utf-8
# DeepSeek-style Mixture of Experts with Shared + Routed Experts
# Reference: DeepSeek-MoE paper, LIMoE paper

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class DeepSeekMoEConfig:
    """Configuration for DeepSeek-style MoE"""

    hidden_size: int = 768
    intermediate_size: int = 3072

    # Shared experts (always active)
    num_shared_experts: int = 2

    # Routed experts
    num_routed_experts: int = 64
    num_experts_per_tok: int = 2  # Top-k routing

    # Routing configuration
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"  # "softmax" or "sigmoid"

    # Auxiliary loss weights
    load_balance_weight: float = 0.01
    z_loss_weight: float = 0.001
    mi_loss_weight: float = 0.0  # LiMoE-style modality importance loss (0 = disabled)

    # Activation
    hidden_act: str = "silu"

    # Dropout
    expert_dropout: float = 0.0


class ExpertMLP(nn.Module):
    """Single Expert FFN"""

    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif hidden_act == "relu":
            self.act_fn = nn.ReLU()
        else:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU-style activation
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekRouter(nn.Module):
    """
    DeepSeek-style Router with Top-k selection.
    Only routes to routed experts (not shared experts).
    """

    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_routed_experts

        # Router linear layer
        self.gate = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)

        # Initialize gate weights
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            topk_idx: [batch_size * seq_len, top_k] - indices of selected experts
            topk_weight: [batch_size * seq_len, top_k] - weights for selected experts
            router_logits: [batch_size * seq_len, num_experts] - raw logits for aux loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Flatten to [B*S, H]
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Compute router logits
        router_logits = self.gate(hidden_states_flat)  # [B*S, num_experts]

        # Compute routing probabilities
        if self.config.scoring_func == "softmax":
            routing_weights = F.softmax(router_logits, dim=-1)
        else:  # sigmoid
            routing_weights = torch.sigmoid(router_logits)

        # Select top-k experts
        topk_weight, topk_idx = torch.topk(routing_weights, k=self.top_k, dim=-1)

        # Normalize top-k weights to sum to 1
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-8)

        # Apply scaling factor
        topk_weight = topk_weight * self.config.routed_scaling_factor

        return topk_idx, topk_weight, router_logits


class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek-style MoE Layer with:
    - Shared Experts: Always active for every token
    - Routed Experts: Selected via Top-k routing

    Tensor Flow:
    Input: [batch_size, seq_len, hidden_size]
    Shared Output: [batch_size, seq_len, hidden_size] (from all shared experts)
    Routed Output: [batch_size, seq_len, hidden_size] (from top-k routed experts)
    Final Output: Shared + Routed outputs
    """

    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList(
            [
                ExpertMLP(
                    config.hidden_size, config.intermediate_size, config.hidden_act
                )
                for _ in range(config.num_shared_experts)
            ]
        )

        # Routed experts
        self.routed_experts = nn.ModuleList(
            [
                ExpertMLP(
                    config.hidden_size, config.intermediate_size, config.hidden_act
                )
                for _ in range(config.num_routed_experts)
            ]
        )

        # Router (only for routed experts)
        self.router = DeepSeekRouter(config)

        # Dropout
        self.expert_dropout = (
            nn.Dropout(config.expert_dropout) if config.expert_dropout > 0 else None
        )

        # Store for aux loss computation
        self._router_logits = None
        self._routing_weights = None
        self._topk_indices = None
        self._modality_indices = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            modality_indices: [batch_size, seq_len] - 0 for image, 1 for text (for MI-Loss)

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Store modality indices for MI-Loss
        self._modality_indices = modality_indices

        # === Shared Experts (always active) ===
        shared_output = torch.zeros_like(hidden_states)
        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(hidden_states)

        # Average over shared experts
        shared_output = shared_output / len(self.shared_experts)

        # === Routed Experts ===
        # Get routing decisions
        topk_idx, topk_weight, router_logits = self.router(hidden_states)

        # Store for aux loss
        self._router_logits = router_logits
        self._routing_weights = topk_weight  # Use actual routing weights, not softmax
        self._topk_indices = topk_idx

        # Compute routed expert outputs
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [B*S, H]
        routed_output_flat = torch.zeros_like(hidden_states_flat)  # [B*S, H]

        # Efficient batched expert computation
        for expert_idx in range(self.config.num_routed_experts):
            # Find tokens routed to this expert
            # topk_idx: [B*S, top_k]
            expert_mask = (topk_idx == expert_idx).any(dim=-1)  # [B*S]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = hidden_states_flat[expert_mask]  # [num_tokens, H]

                # Compute expert output
                expert_output = self.routed_experts[expert_idx](
                    expert_input
                )  # [num_tokens, H]

                # Get weights for this expert
                # For each token, find the weight if this expert was selected
                token_indices = torch.where(expert_mask)[0]  # [num_tokens]

                for i, token_idx in enumerate(token_indices):
                    # Find which position in topk_idx contains this expert
                    expert_positions = topk_idx[token_idx] == expert_idx
                    if expert_positions.any():
                        weight = topk_weight[token_idx][expert_positions].sum()
                        routed_output_flat[token_idx] += weight * expert_output[i]

        # Reshape routed output
        routed_output = routed_output_flat.view(batch_size, seq_len, hidden_size)

        # Apply dropout if configured
        if self.expert_dropout is not None:
            routed_output = self.expert_dropout(routed_output)

        # Combine shared and routed outputs
        output = shared_output + routed_output

        return output

    def compute_load_balance_loss(self) -> torch.Tensor:
        """
        Compute load balancing loss for routed experts only.
        Uses Switch Transformer style: sum(f_i * P_i) where:
        - f_i: fraction of tokens routed to expert i
        - P_i: average routing probability for expert i

        Only applies to ROUTED experts (not shared).
        """
        if self._router_logits is None:
            return torch.tensor(0.0)

        routing_probs = F.softmax(self._router_logits, dim=-1)  # [B*S, num_experts]
        num_tokens = routing_probs.shape[0]
        num_experts = self.config.num_routed_experts

        # Average routing probability per expert
        avg_prob = routing_probs.mean(dim=0)  # [num_experts]

        # Fraction of tokens assigned to each expert (using hard assignment)
        _, topk_idx = torch.topk(
            routing_probs, k=self.config.num_experts_per_tok, dim=-1
        )

        # Count tokens per expert
        expert_counts = torch.zeros(num_experts, device=routing_probs.device)
        for expert_idx in range(num_experts):
            expert_counts[expert_idx] = (topk_idx == expert_idx).sum().float()

        # Normalize to get fraction
        token_fraction = expert_counts / (
            num_tokens * self.config.num_experts_per_tok + 1e-8
        )

        # Load balancing loss
        load_balance_loss = num_experts * (token_fraction * avg_prob).sum()

        return load_balance_loss * self.config.load_balance_weight

    def compute_z_loss(self) -> torch.Tensor:
        """
        Z-loss for router logit stability.
        Prevents router logits from becoming too large.
        L_z = mean(log(sum(exp(logits)))^2)
        """
        if self._router_logits is None:
            return torch.tensor(0.0)

        log_sum_exp = torch.logsumexp(self._router_logits, dim=-1)  # [B*S]
        z_loss = torch.mean(log_sum_exp**2)

        return z_loss * self.config.z_loss_weight

    def compute_mi_loss(self) -> torch.Tensor:
        """
        LiMoE-style Modality Importance Loss.

        Encourages balanced expert usage across modalities by penalizing
        experts that are dominated by one modality.

        From LiMoE paper: "Limiting Modality Experts" to prevent modality collapse.

        Two variants:
        1. Entropy-based: Encourage each expert to be used equally by both modalities
        2. Balance-based: Ensure visual and text tokens use similar expert distributions

        We implement the balance-based approach:
        L_mi = KL(P_vision || P_text) + KL(P_text || P_vision)

        Where P_vision[i] = fraction of vision tokens routed to expert i
              P_text[i] = fraction of text tokens routed to expert i
        """
        if self._modality_indices is None or self._routing_weights is None:
            return torch.tensor(0.0, device=self._routing_weights.device)

        if self._topk_indices is None:
            return torch.tensor(0.0, device=self._routing_weights.device)

        device = self._routing_weights.device
        num_experts = self.config.num_routed_experts

        # Flatten modality indices
        modality_flat = self._modality_indices.view(-1)  # [B*S]
        topk_indices = self._topk_indices  # [B*S, top_k]
        topk_weights = self._routing_weights  # [B*S, top_k]

        # Separate by modality
        vision_mask = modality_flat == 0  # Vision tokens
        text_mask = modality_flat == 1  # Text tokens

        num_vision = vision_mask.sum().float()
        num_text = text_mask.sum().float()

        if num_vision == 0 or num_text == 0:
            return torch.tensor(0.0, device=device)

        # Compute expert usage distribution per modality
        # P_vision[i] = sum of weights assigned to expert i by vision tokens
        # P_text[i] = sum of weights assigned to expert i by text tokens

        vision_expert_usage = torch.zeros(num_experts, device=device)
        text_expert_usage = torch.zeros(num_experts, device=device)

        # For vision tokens
        if vision_mask.any():
            vision_indices = topk_indices[vision_mask]  # [N_vision, top_k]
            vision_weights = topk_weights[vision_mask]  # [N_vision, top_k]

            for k in range(self.config.num_experts_per_tok):
                vision_expert_usage.scatter_add_(
                    0, vision_indices[:, k], vision_weights[:, k]
                )

        # For text tokens
        if text_mask.any():
            text_indices = topk_indices[text_mask]  # [N_text, top_k]
            text_weights = topk_weights[text_mask]  # [N_text, top_k]

            for k in range(self.config.num_experts_per_tok):
                text_expert_usage.scatter_add_(
                    0, text_indices[:, k], text_weights[:, k]
                )

        # Normalize to get probability distributions
        vision_prob = vision_expert_usage / (vision_expert_usage.sum() + 1e-8)
        text_prob = text_expert_usage / (text_expert_usage.sum() + 1e-8)

        # Add small epsilon for numerical stability
        eps = 1e-8
        vision_prob = vision_prob + eps
        text_prob = text_prob + eps

        # Renormalize
        vision_prob = vision_prob / vision_prob.sum()
        text_prob = text_prob / text_prob.sum()

        # Symmetric KL divergence: encourages similar expert usage across modalities
        # This prevents one modality from "dominating" certain experts
        kl_v_to_t = F.kl_div(text_prob.log(), vision_prob, reduction="sum")
        kl_t_to_v = F.kl_div(vision_prob.log(), text_prob, reduction="sum")

        mi_loss = (kl_v_to_t + kl_t_to_v) / 2

        return mi_loss * self.config.mi_loss_weight

    def get_aux_losses(self) -> Dict[str, torch.Tensor]:
        """Get all auxiliary losses"""
        return {
            "load_balance_loss": self.compute_load_balance_loss(),
            "z_loss": self.compute_z_loss(),
            "mi_loss": self.compute_mi_loss(),
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DeepSeek-style MoE Layer")
    print("=" * 60)

    # Configuration
    config = DeepSeekMoEConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_shared_experts=2,
        num_routed_experts=8,  # Using 8 for testing (64 would be too heavy)
        num_experts_per_tok=2,
    )

    # Create MoE layer
    moe_layer = DeepSeekMoELayer(config)

    # Test input
    batch_size = 2
    seq_len = 96  # 64 visual + 32 text tokens

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Modality indices: 0 for image, 1 for text
    modality_indices = torch.cat(
        [
            torch.zeros(batch_size, 64, dtype=torch.long),  # 64 image tokens
            torch.ones(batch_size, 32, dtype=torch.long),  # 32 text tokens
        ],
        dim=1,
    )

    print(f"Input shape: {hidden_states.shape}")
    print(f"Modality indices shape: {modality_indices.shape}")

    # Forward pass
    output = moe_layer(hidden_states, modality_indices)

    print(f"Output shape: {output.shape}")

    # Get auxiliary losses
    aux_losses = moe_layer.get_aux_losses()
    print(f"\nAuxiliary Losses:")
    for name, loss in aux_losses.items():
        print(f"  {name}: {loss.item():.6f}")

    print("\n✓ DeepSeek MoE Layer test passed!")

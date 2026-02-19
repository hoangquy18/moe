# coding=utf-8
# DeepSeek-style Mixture of Experts with Shared + Routed Experts
# Reference: DeepSeek-MoE paper, LIMoE paper, DeepSeek-V3 paper

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class DeepSeekMoEConfig:
    """
    Configuration for DeepSeek-style MoE.

    Load Balancing Modes
    --------------------
    Mode 1 — Auxiliary Loss (default, use_aux_free_balancing=False):
        Adds load_balance_loss + z_loss to the training objective.
        Standard approach from Switch Transformer / DeepSeek-MoE paper.

    Mode 2 — Auxiliary-Free (use_aux_free_balancing=True, DeepSeek-V3):
        Uses per-expert bias b_i that is updated WITHOUT gradients.
        Selection: top-k(score_i + b_i)
        Weight:    score_i  (no bias, pure routing quality)
        Bias update: b_i += γ * sign(target_load - actual_load)
        Overloaded experts get lower bias → less likely to be chosen.
        Eliminates the need for load_balance_weight tuning.

    Routing Modes
    -------------
    Token Choice (default): Each token selects top-k experts.
    Expert Choice (routing_mode="expert_choice"):
        Each expert selects top-c tokens (c = capacity).
        Guarantees perfect load balance. No token dropping needed.
        Note: some tokens may not be routed to any expert.

    Expert Capacity
    ---------------
    capacity_factor > 0 enables per-expert token limits.
    Overflow tokens are dropped (output = 0) when drop_tokens=True.
    capacity = max(capacity_factor * total_tokens * top_k / num_experts, min_capacity)
    Typical values: 1.0 (tight), 1.25 (5% slack), 2.0 (lenient).
    """

    hidden_size: int = 768
    intermediate_size: int = 3072

    # Shared experts (always active, no routing)
    num_shared_experts: int = 2

    # Routed experts
    num_routed_experts: int = 64
    num_experts_per_tok: int = 2  # Top-k routing (token choice)

    # Routing mode
    routing_mode: str = "token_choice"  # "token_choice" | "expert_choice"

    # Routing configuration
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"  # "softmax" | "sigmoid"

    # ── Expert Capacity (token dropping) ──────────────────────────────────────
    # capacity_factor = 0.0 → no limit (default, backward-compatible)
    # capacity_factor = 1.25 → each expert handles at most 25% extra tokens
    capacity_factor: float = 0.0
    drop_tokens: bool = True  # True = drop overflow tokens; False = ignore limit
    min_capacity: int = 4  # Minimum tokens an expert must be able to receive

    # ── Auxiliary-Free Load Balancing (DeepSeek-V3) ───────────────────────────
    use_aux_free_balancing: bool = False
    aux_free_bias_update_rate: float = 0.001  # γ (gamma) — step size for bias update

    # ── Auxiliary Loss Weights (used when use_aux_free_balancing=False) ───────
    load_balance_weight: float = 0.01
    z_loss_weight: float = 0.001
    mi_loss_weight: float = 0.0  # LiMoE-style modality importance loss (0 = disabled)

    # Activation
    hidden_act: str = "silu"

    # Dropout
    expert_dropout: float = 0.0


class ExpertMLP(nn.Module):
    """Single Expert FFN (SwiGLU)"""

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
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekRouter(nn.Module):
    """
    DeepSeek-style Router with Top-k selection.

    Supports two load balancing modes:
      1. Auxiliary loss  (use_aux_free_balancing=False) — default
      2. Auxiliary-free  (use_aux_free_balancing=True)  — DeepSeek-V3

    In aux-free mode:
      - A per-expert bias is maintained as a non-gradient buffer.
      - Selection: top-k(score + bias)  ← bias steers load
      - Weights  : score at selected positions (no bias influence on weights)
      - Bias update: called automatically during training forward pass.
    """

    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_routed_experts

        # Router linear layer (no bias — logit bias handled separately)
        self.gate = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))

        # Per-expert bias for auxiliary-free load balancing (non-gradient buffer)
        if config.use_aux_free_balancing:
            self.register_buffer("expert_bias", torch.zeros(config.num_routed_experts))

    def _compute_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute routing scores from raw logits."""
        if self.config.scoring_func == "softmax":
            return F.softmax(router_logits, dim=-1)
        else:  # sigmoid
            return torch.sigmoid(router_logits)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            topk_idx:     [B*S, top_k] — indices of selected experts
            topk_weight:  [B*S, top_k] — weights for selected experts (no bias)
            router_logits:[B*S, num_experts] — raw logits for aux loss computation
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [B*S, H]

        # Raw router logits (pure learned routing signal, no bias)
        router_logits = self.gate(hidden_states_flat)  # [B*S, num_experts]

        # Routing scores (used for weight computation — always without bias)
        routing_scores = self._compute_scores(router_logits)  # [B*S, num_experts]

        # Selection scores (with bias for aux-free balancing)
        if self.config.use_aux_free_balancing:
            # Bias shifts selection toward underloaded experts.
            # The bias does NOT affect final token weights — only expert selection.
            selection_scores = routing_scores + self.expert_bias.unsqueeze(0)
        else:
            selection_scores = routing_scores

        # Top-k selection
        _, topk_idx = torch.topk(selection_scores, k=self.top_k, dim=-1)  # [B*S, top_k]

        # Gather actual weights at selected positions (unbiased)
        topk_weight = routing_scores.gather(-1, topk_idx)  # [B*S, top_k]

        # Normalize weights to sum to 1 per token
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-8)

        # Apply scaling factor
        topk_weight = topk_weight * self.config.routed_scaling_factor

        # Update expert bias during training (no gradient)
        if self.config.use_aux_free_balancing and self.training:
            self._update_expert_bias(topk_idx)

        return topk_idx, topk_weight, router_logits

    @torch.no_grad()
    def _update_expert_bias(self, topk_idx: torch.Tensor) -> None:
        """
        Update per-expert bias based on observed token load (DeepSeek-V3).

        Rule:  b_i ← b_i + γ · sign(target_load − actual_load)

        - Overloaded  experts: bias decreases → less likely to be selected next step
        - Underloaded experts: bias increases → more likely to be selected next step

        This is a running heuristic, NOT a gradient update.
        """
        num_tokens = topk_idx.shape[0]

        # Vectorized token count per expert using bincount
        expert_counts = torch.bincount(
            topk_idx.reshape(-1),
            minlength=self.num_experts,
        ).float()  # [num_experts]

        # Ideal uniform load
        target_load = (num_tokens * self.top_k) / self.num_experts

        # Update: sign gives -1 (overloaded) or +1 (underloaded) or 0 (balanced)
        self.expert_bias.add_(
            self.config.aux_free_bias_update_rate
            * torch.sign(target_load - expert_counts)
        )

    def get_expert_bias(self) -> Optional[torch.Tensor]:
        """Return current expert bias (only for aux-free mode)."""
        if self.config.use_aux_free_balancing:
            return self.expert_bias.clone()
        return None


class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek-style MoE Layer.

    Architecture
    ============
    - Shared Experts  (num_shared_experts): Always active for every token.
    - Routed Experts  (num_routed_experts): Selected via routing (token-choice or expert-choice).

    Tensor Flow
    ===========
    Input:        [B, S, H]
    Shared path:  [B, S, H] → sum(shared_experts(·)) / N_shared
    Routed path:  [B, S, H] → router → dispatch → experts → gather → [B, S, H]
    Output:       shared + routed
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

        # Cached state for auxiliary loss computation (populated during forward)
        self._router_logits: Optional[torch.Tensor] = None
        self._routing_weights: Optional[torch.Tensor] = None
        self._topk_indices: Optional[torch.Tensor] = None
        self._modality_indices: Optional[torch.Tensor] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Expert Dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_expert_capacity(self, num_tokens: int) -> int:
        """
        Compute per-expert token capacity.

        Returns 0 when capacity limiting is disabled.
        """
        if self.config.capacity_factor <= 0 or not self.config.drop_tokens:
            return 0  # No limit

        capacity = int(
            self.config.capacity_factor
            * num_tokens
            * self.config.num_experts_per_tok
            / self.config.num_routed_experts
        )
        return max(capacity, self.config.min_capacity)

    def _dispatch_token_choice(
        self,
        hidden_states_flat: torch.Tensor,  # [num_tokens, H]
        topk_idx: torch.Tensor,  # [num_tokens, top_k]
        topk_weight: torch.Tensor,  # [num_tokens, top_k]
    ) -> torch.Tensor:
        """
        Vectorized token-choice expert dispatch.

        For each expert:
          1. Find which tokens are routed to it.
          2. Apply capacity limit (drop overflow tokens if configured).
          3. Gather those tokens, run expert forward.
          4. Compute per-token weights (vectorized, NO inner Python for-loop).
          5. Scatter weighted outputs back.

        Eliminates the O(num_tokens) Python inner loop from the original code.
        """
        num_tokens, hidden_size = hidden_states_flat.shape
        num_experts = self.config.num_routed_experts
        capacity = self._compute_expert_capacity(num_tokens)

        routed_output_flat = torch.zeros_like(hidden_states_flat)

        for expert_idx in range(num_experts):
            # ── Which tokens are routed to this expert? ──
            # token_expert_mask: [num_tokens, top_k] — True where this expert is selected
            token_expert_mask = topk_idx == expert_idx  # [num_tokens, top_k]
            # token_mask: [num_tokens] — True if any k-slot selected this expert
            token_mask = token_expert_mask.any(dim=-1)  # [num_tokens]

            if not token_mask.any():
                continue

            # ── Apply capacity limit (token dropping) ──
            if capacity > 0:
                selected_indices = token_mask.nonzero(as_tuple=False).squeeze(-1)
                if selected_indices.shape[0] > capacity:
                    # Drop excess tokens (first-come-first-served by position index)
                    overflow_indices = selected_indices[capacity:]
                    token_mask = token_mask.clone()
                    token_mask[overflow_indices] = False
                    token_expert_mask = token_expert_mask.clone()
                    token_expert_mask[overflow_indices] = False

            # ── Gather inputs ──
            expert_input = hidden_states_flat[token_mask]  # [N_sel, H]

            # ── Run expert ──
            expert_output = self.routed_experts[expert_idx](expert_input)  # [N_sel, H]

            # ── Compute weights — fully vectorized, no inner for-loop ──
            # Sum weights across k-slots (typically only 1 slot per expert per token)
            token_weights = (topk_weight * token_expert_mask.float()).sum(
                dim=-1
            )  # [num_tokens]
            selected_weights = token_weights[token_mask]  # [N_sel]

            # ── Scatter weighted output back ──
            routed_output_flat[token_mask] = (
                routed_output_flat[token_mask]
                + selected_weights.unsqueeze(-1) * expert_output
            )

        return routed_output_flat

    def _dispatch_expert_choice(
        self,
        hidden_states_flat: torch.Tensor,  # [num_tokens, H]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing dispatch (alternative to token-choice).

        Each expert selects the top-c tokens it wants to process.
        c = capacity_factor * num_tokens / num_experts  (must set capacity_factor > 0)

        Advantages over token-choice:
          - Guarantees perfectly uniform expert load (no token dropping needed).
          - Naturally avoids expert collapse.
        Disadvantage:
          - Some tokens may not be routed to ANY expert (they only get shared expert output).
          - Requires capacity_factor to be set.

        Returns:
            routed_output_flat: [num_tokens, H]
            topk_idx:           [num_tokens, top_k]  (fake, for aux loss compat)
            topk_weight:        [num_tokens, top_k]
        """
        num_tokens, hidden_size = hidden_states_flat.shape
        num_experts = self.config.num_routed_experts
        device = hidden_states_flat.device

        # Capacity: how many tokens each expert processes
        if self.config.capacity_factor > 0:
            capacity = max(
                int(self.config.capacity_factor * num_tokens / num_experts),
                self.config.min_capacity,
            )
        else:
            # Default: same total assignments as token-choice top-k
            capacity = max(
                int(num_tokens * self.config.num_experts_per_tok / num_experts),
                self.config.min_capacity,
            )

        # Compute routing scores for all (token, expert) pairs
        router_logits_2d = self.router.gate(
            hidden_states_flat
        )  # [num_tokens, num_experts]
        routing_scores = self.router._compute_scores(
            router_logits_2d
        )  # [num_tokens, num_experts]

        routed_output_flat = torch.zeros_like(hidden_states_flat)

        # Build fake topk_idx / topk_weight for aux loss compatibility
        # We approximate: for each token, record which experts chose it
        token_expert_assignment = torch.full(
            (num_tokens, self.config.num_experts_per_tok),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        token_expert_weights = torch.zeros(
            num_tokens, self.config.num_experts_per_tok, device=device
        )

        for expert_idx in range(num_experts):
            # Expert selects top-c tokens by routing score
            scores_for_expert = routing_scores[:, expert_idx]  # [num_tokens]
            _, chosen_token_idx = torch.topk(
                scores_for_expert, k=capacity, dim=0
            )  # [capacity]

            chosen_weights = scores_for_expert[chosen_token_idx]  # [capacity]
            expert_input = hidden_states_flat[chosen_token_idx]  # [capacity, H]
            expert_output = self.routed_experts[expert_idx](expert_input)

            # Accumulate (tokens may be chosen by multiple experts)
            routed_output_flat.scatter_add_(
                0,
                chosen_token_idx.unsqueeze(-1).expand(-1, hidden_size),
                chosen_weights.unsqueeze(-1) * expert_output,
            )

            # Record assignment for aux loss (fill first available slot)
            for slot in range(self.config.num_experts_per_tok):
                empty_slots = token_expert_assignment[chosen_token_idx, slot] == -1
                if empty_slots.any():
                    mask_indices = chosen_token_idx[empty_slots]
                    token_expert_assignment[mask_indices, slot] = expert_idx
                    token_expert_weights[mask_indices, slot] = chosen_weights[
                        empty_slots
                    ]
                    break

        # Fill unassigned slots with 0 for compat
        token_expert_assignment[token_expert_assignment == -1] = 0

        return routed_output_flat, token_expert_assignment, token_expert_weights

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:    [batch_size, seq_len, hidden_size]
            modality_indices: [batch_size, seq_len] — 0=image, 1=text (for MI-Loss)

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        self._modality_indices = modality_indices

        # ── Shared Experts (always active, no routing) ──
        shared_output = torch.zeros_like(hidden_states)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(hidden_states)
        shared_output = shared_output / len(self.shared_experts)

        # ── Routed Experts ──
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [B*S, H]

        if self.config.routing_mode == "expert_choice":
            # Expert-choice: each expert picks its top-c tokens
            routed_output_flat, topk_idx, topk_weight = self._dispatch_expert_choice(
                hidden_states_flat
            )
            # Also get router logits for z-loss / mi-loss
            router_logits = self.router.gate(hidden_states_flat)
        else:
            # Token-choice (default): each token picks top-k experts
            topk_idx, topk_weight, router_logits = self.router(hidden_states)
            routed_output_flat = self._dispatch_token_choice(
                hidden_states_flat, topk_idx, topk_weight
            )

        # Cache for auxiliary loss computation
        self._router_logits = router_logits
        self._routing_weights = topk_weight
        self._topk_indices = topk_idx

        # Reshape routed output
        routed_output = routed_output_flat.view(batch_size, seq_len, hidden_size)

        # Dropout
        if self.expert_dropout is not None:
            routed_output = self.expert_dropout(routed_output)

        return shared_output + routed_output

    # ──────────────────────────────────────────────────────────────────────────
    # Auxiliary Losses
    # ──────────────────────────────────────────────────────────────────────────

    def compute_load_balance_loss(self) -> torch.Tensor:
        """
        Switch Transformer / DeepSeek-MoE load balancing loss.

        L_lb = num_experts · Σ_i ( f_i · P_i )

        where:
          f_i = fraction of tokens routed to expert i  (hard assignment)
          P_i = mean routing probability to expert i   (soft, differentiable)

        Only active when use_aux_free_balancing=False.
        Fully vectorized using torch.bincount — no Python for-loop.
        """
        if self._router_logits is None:
            return torch.tensor(0.0)

        if self.config.use_aux_free_balancing:
            # Bias mechanism handles balancing — no aux loss needed
            return torch.tensor(0.0, device=self._router_logits.device)

        routing_probs = F.softmax(self._router_logits, dim=-1)  # [B*S, num_experts]
        num_tokens = routing_probs.shape[0]
        num_experts = self.config.num_routed_experts

        # P_i: average routing probability per expert (differentiable)
        avg_prob = routing_probs.mean(dim=0)  # [num_experts]

        # f_i: fraction of tokens assigned to each expert (hard, stop-grad)
        # Vectorized: use topk on routing_probs for hard assignment
        _, hard_topk_idx = torch.topk(
            routing_probs, k=self.config.num_experts_per_tok, dim=-1
        )  # [B*S, top_k]

        # bincount is faster and cleaner than a for-loop
        expert_counts = torch.bincount(
            hard_topk_idx.reshape(-1),
            minlength=num_experts,
        ).float()  # [num_experts]

        token_fraction = expert_counts / (
            num_tokens * self.config.num_experts_per_tok + 1e-8
        )

        # Load balancing loss
        load_balance_loss = num_experts * (token_fraction * avg_prob).sum()

        return load_balance_loss * self.config.load_balance_weight

    def compute_z_loss(self) -> torch.Tensor:
        """
        Router logit stability (z-loss).

        L_z = mean( log(Σ_i exp(logit_i))² )

        Prevents router logits from becoming too large.
        """
        if self._router_logits is None:
            return torch.tensor(0.0)

        log_sum_exp = torch.logsumexp(self._router_logits, dim=-1)  # [B*S]
        z_loss = torch.mean(log_sum_exp**2)

        return z_loss * self.config.z_loss_weight

    def compute_mi_loss(self) -> torch.Tensor:
        """
        LiMoE-style Modality Importance (MI) Loss.

        Encourages balanced expert usage across modalities to prevent
        "modality collapse" (experts being dominated by one modality).

        Implementation: symmetric KL divergence between per-modality
        expert usage distributions:

            L_mi = [ KL(P_vision ‖ P_text) + KL(P_text ‖ P_vision) ] / 2

        where P_vision[i] = fraction of vision-token routing weight to expert i
              P_text[i]   = fraction of text-token routing weight to expert i
        """
        if self._modality_indices is None:
            return torch.tensor(0.0)

        if self._routing_weights is None or self._topk_indices is None:
            return torch.tensor(0.0)

        device = self._routing_weights.device
        num_experts = self.config.num_routed_experts

        # Flatten modality indices to match [B*S] layout
        modality_flat = self._modality_indices.view(-1)  # [B*S]
        topk_indices = self._topk_indices  # [B*S, top_k]
        topk_weights = self._routing_weights  # [B*S, top_k]

        vision_mask = modality_flat == 0
        text_mask = modality_flat == 1

        if not vision_mask.any() or not text_mask.any():
            return torch.tensor(0.0, device=device)

        def _expert_usage(mask: torch.Tensor) -> torch.Tensor:
            """Sum routing weights per expert for tokens in `mask`."""
            indices = topk_indices[mask]  # [N, top_k]
            weights = topk_weights[mask]  # [N, top_k]
            usage = torch.zeros(num_experts, device=device)
            # Vectorized scatter over all k-slots
            usage.scatter_add_(0, indices.reshape(-1), weights.reshape(-1))
            return usage

        vision_usage = _expert_usage(vision_mask)
        text_usage = _expert_usage(text_mask)

        # Normalize to probability distributions
        eps = 1e-8
        vision_prob = (vision_usage + eps) / (vision_usage.sum() + eps * num_experts)
        text_prob = (text_usage + eps) / (text_usage.sum() + eps * num_experts)

        # Symmetric KL divergence (Jensen-Shannon-like)
        kl_v_to_t = F.kl_div(text_prob.log(), vision_prob, reduction="sum")
        kl_t_to_v = F.kl_div(vision_prob.log(), text_prob, reduction="sum")
        mi_loss = (kl_v_to_t + kl_t_to_v) / 2

        return mi_loss * self.config.mi_loss_weight

    def get_aux_losses(self) -> Dict[str, torch.Tensor]:
        """Return all auxiliary losses as a dict."""
        return {
            "load_balance_loss": self.compute_load_balance_loss(),
            "z_loss": self.compute_z_loss(),
            "mi_loss": self.compute_mi_loss(),
        }

    def get_routing_stats(self) -> Dict[str, torch.Tensor]:
        """
        Return per-expert routing statistics for logging/debugging.

        Useful for monitoring load balance during training.
        """
        if self._topk_indices is None:
            return {}

        num_tokens = self._topk_indices.shape[0]
        num_experts = self.config.num_routed_experts

        expert_counts = torch.bincount(
            self._topk_indices.reshape(-1), minlength=num_experts
        ).float()

        return {
            "expert_counts": expert_counts,
            "expert_load_fraction": expert_counts
            / (num_tokens * self.config.num_experts_per_tok),
            "load_std": expert_counts.std(),
            "load_max": expert_counts.max(),
            "load_min": expert_counts.min(),
            "expert_bias": self.router.get_expert_bias(),
        }


if __name__ == "__main__":
    print("=" * 70)
    print("Testing DeepSeek-style MoE Layer — Enhanced Version")
    print("=" * 70)

    batch_size = 2
    seq_len = 96  # 64 visual + 32 text tokens

    modality_indices = torch.cat(
        [
            torch.zeros(batch_size, 64, dtype=torch.long),
            torch.ones(batch_size, 32, dtype=torch.long),
        ],
        dim=1,
    )

    # ── Test 1: Auxiliary Loss mode (default) ──
    print("\n[Test 1] Auxiliary Loss mode (token-choice, no capacity)")
    config1 = DeepSeekMoEConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_shared_experts=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        load_balance_weight=0.01,
        z_loss_weight=0.001,
        mi_loss_weight=0.01,
    )
    moe1 = DeepSeekMoELayer(config1)
    x = torch.randn(batch_size, seq_len, config1.hidden_size)
    out1 = moe1(x, modality_indices)
    print(f"  Input:  {x.shape}  →  Output: {out1.shape}")
    losses1 = moe1.get_aux_losses()
    for k, v in losses1.items():
        print(f"  {k}: {v.item():.6f}")

    # ── Test 2: Auxiliary-Free mode (DeepSeek-V3) ──
    print("\n[Test 2] Auxiliary-Free mode (DeepSeek-V3 bias)")
    config2 = DeepSeekMoEConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_shared_experts=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        use_aux_free_balancing=True,
        aux_free_bias_update_rate=0.001,
        mi_loss_weight=0.01,
    )
    moe2 = DeepSeekMoELayer(config2)
    moe2.train()
    out2 = moe2(x, modality_indices)
    print(f"  Input:  {x.shape}  →  Output: {out2.shape}")
    print(f"  Expert bias (after 1 step): {moe2.router.expert_bias.tolist()}")
    losses2 = moe2.get_aux_losses()
    print(
        f"  load_balance_loss (should be 0): {losses2['load_balance_loss'].item():.6f}"
    )
    print(f"  z_loss: {losses2['z_loss'].item():.6f}")

    # ── Test 3: Expert Capacity + Token Dropping ──
    print("\n[Test 3] Token-choice with capacity (capacity_factor=1.25)")
    config3 = DeepSeekMoEConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_shared_experts=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        capacity_factor=1.25,
        drop_tokens=True,
    )
    moe3 = DeepSeekMoELayer(config3)
    out3 = moe3(x, modality_indices)
    print(f"  Input:  {x.shape}  →  Output: {out3.shape}")
    stats = moe3.get_routing_stats()
    print(f"  Expert counts: {stats['expert_counts'].int().tolist()}")
    print(f"  Load std: {stats['load_std'].item():.4f}")

    # ── Test 4: Expert-Choice Routing ──
    print("\n[Test 4] Expert-choice routing")
    config4 = DeepSeekMoEConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_shared_experts=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        routing_mode="expert_choice",
        capacity_factor=1.0,
    )
    moe4 = DeepSeekMoELayer(config4)
    out4 = moe4(x, modality_indices)
    print(f"  Input:  {x.shape}  →  Output: {out4.shape}")

    print("\n✓ All tests passed!")

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Helper Functions ---
def entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Calculates entropy H(p) = -sum p log(p)."""
    log_probs = torch.log(probs + eps)
    return -torch.sum(probs * log_probs, dim=dim)


def squared_coefficient_of_variation(
    x: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Calculates (std(x) / mean(x))^2. Expects x to be 1D."""
    if x.numel() <= 1:  # std is not well-defined for 0 or 1 element
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    mean_x = torch.mean(x)
    std_x = torch.std(x, unbiased=False)  # Population std, as typically used
    return (std_x / (mean_x + eps)) ** 2


# --- Individual Auxiliary Loss Functions ---


def importance_loss(
    gating_probs: torch.Tensor, num_experts: int, eps: float = 1e-8
) -> torch.Tensor:
    """
    Importance loss from V-MoE / Switch Transformers, also in LIMoE Appendix B.1.
    Ω_imp(X) = (std(imp(X)) / mean(imp(X)))^2
    where imp_e(X) = sum_{x in X} g(x)_e

    Args:
        gating_probs (torch.Tensor): Probabilities from the router softmax.
                                     Shape: (num_tokens, num_experts).
        num_experts (int): Total number of experts.
        eps (float): Epsilon for numerical stability.
    Returns:
        torch.Tensor: Scalar importance loss.
    """
    if gating_probs.shape[0] == 0:  # No tokens
        return torch.tensor(0.0, device=gating_probs.device, dtype=gating_probs.dtype)

    # imp_per_expert: sum of gating probabilities for each expert
    # Shape: (num_experts,)
    imp_per_expert = torch.sum(gating_probs, dim=0)
    return squared_coefficient_of_variation(imp_per_expert, eps)


def load_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    k_selected_experts: int,  # K for Top-K routing
    noise_stddev_routing: float,  # sigma for adding noise to logits (e.g., 1.0/num_experts)
    noise_stddev_cdf: float,  # sigma for the CDF calculation (e.g., 1.0/num_experts)
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Load loss from LIMoE Appendix B.1, inspired by Shazeer et al. 2017.
    This version matches the description with noise and Gaussian CDF.
    Ω_load(X) = (std(load(X)) / mean(load(X)))^2
    where load_e(X) = sum_{x in X} p_e(x)
    and p_e(x) = 1 - Φ((η_k - (Wx)_e) / σ_cdf)

    Args:
        router_logits (torch.Tensor): Raw logits from the router Wx.
                                     Shape: (num_tokens, num_experts).
        num_experts (int): Total number of experts.
        k_selected_experts (int): The 'K' in Top-K routing.
        noise_stddev_routing (float): Stddev for the Gaussian noise added to logits
                                      before finding η_k.
        noise_stddev_cdf (float): The 'σ' in the denominator of the CDF argument.
        eps (float): Epsilon for numerical stability.
    Returns:
        torch.Tensor: Scalar load loss.
    """
    if router_logits.shape[0] == 0:  # No tokens
        return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

    num_tokens = router_logits.shape[0]

    # Add noise to router logits
    noise = torch.randn_like(router_logits) * noise_stddev_routing
    noisy_logits = router_logits + noise

    # Find η_k: the K-th largest noisy logit for each token
    # top_k returns values and indices. We need the K-th value.
    # Shape: (num_tokens,)
    eta_k = torch.topk(noisy_logits, k_selected_experts, dim=1).values[:, -1]

    # Calculate p_e(x) for each token and expert
    # p_e(x) = 1 - Φ((η_k - (Wx)_e) / σ_cdf)
    # Φ is the CDF of a standard normal distribution N(0,1)
    # Argument for CDF: (η_k.unsqueeze(1) - router_logits) / noise_stddev_cdf
    # unsqueeze eta_k to make it broadcastable with router_logits
    # Shape: (num_tokens, num_experts)
    cdf_arg = (eta_k.unsqueeze(1).expand_as(router_logits) - router_logits) / (
        noise_stddev_cdf + eps
    )

    # Using torch.distributions.Normal for CDF
    # normal_dist.cdf(x) gives P(X <= x) where X ~ N(0,1)
    normal_dist = torch.distributions.Normal(0, 1)
    prob_expert_selected_among_topk = 1.0 - normal_dist.cdf(cdf_arg)  # p_e(x)

    # load_per_expert: sum of p_e(x) for each expert
    # Shape: (num_experts,)
    load_per_expert = torch.sum(prob_expert_selected_among_topk, dim=0)

    return squared_coefficient_of_variation(load_per_expert, eps)


def z_loss_fn(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Z-loss from ST-MoE (Fedus et al. 2022) / GShard (Lepikhin et al. 2021), also in LIMoE Appendix B.1.
    L_zloss(X) = (1/n) * sum_{i=1 to n} (log(sum_{e=1 to E} exp((Wx_i)_e)))^2
               = mean ( (logsumexp( (Wx_i)_e ))^2 )

    Args:
        router_logits (torch.Tensor): Raw logits from the router Wx.
                                     Shape: (num_tokens, num_experts).
    Returns:
        torch.Tensor: Scalar Z-loss.
    """
    if router_logits.shape[0] == 0:  # No tokens
        return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

    log_sum_exp_logits = torch.logsumexp(router_logits, dim=1)  # per token
    return torch.mean(log_sum_exp_logits**2)


def local_entropy_loss_modal(
    gating_probs_modal: torch.Tensor,  # gating_probs for a single modality
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Local entropy loss for tokens of a single modality (Eq. 2).
    Ω_local(Gm) = (1/N_m) * sum_{i=1 to N_m} H(p_m(experts|x_i))

    Args:
        gating_probs_modal (torch.Tensor): Gating probabilities for tokens of one modality.
                                         Shape: (num_tokens_modal, num_experts).
        eps (float): Epsilon for numerical stability in entropy calculation.
    Returns:
        torch.Tensor: Scalar local entropy loss for this modality.
    """
    if gating_probs_modal.shape[0] == 0:
        return torch.tensor(
            0.0, device=gating_probs_modal.device, dtype=gating_probs_modal.dtype
        )

    per_token_entropy = entropy(gating_probs_modal, dim=1, eps=eps)
    return torch.mean(per_token_entropy)


def global_entropy_loss_modal(
    gating_probs_modal: torch.Tensor,  # gating_probs for a single modality
    threshold: float,  # τ_m = log(S_m), where S_m is soft min experts for modality m
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Global entropy loss for tokens of a single modality (Eq. 2, with thresholding).
    Ω_global(Gm) = max(0, τ_m - H(p_m(experts)))
    where p_m(experts) is the average gating distribution for modality m.

    Args:
        gating_probs_modal (torch.Tensor): Gating probabilities for tokens of one modality.
                                         Shape: (num_tokens_modal, num_experts).
        threshold (float): The target minimum entropy τ_m (e.g., log(S_m)).
        eps (float): Epsilon for numerical stability.
    Returns:
        torch.Tensor: Scalar global entropy loss for this modality.
    """
    if gating_probs_modal.shape[0] == 0:
        return torch.tensor(
            0.0, device=gating_probs_modal.device, dtype=gating_probs_modal.dtype
        )

    # p_m(experts): average gating distribution for this modality
    avg_gating_probs_modal = torch.mean(
        gating_probs_modal, dim=0
    )  # Shape: (num_experts,)
    current_global_entropy = entropy(avg_gating_probs_modal, dim=0, eps=eps)

    # Loss = max(0, target_entropy_threshold - current_global_entropy)
    # The paper says Ω_global(Gm) = -H(p_m(experts)), and then applies threshold:
    # global_loss = max(0, τ + Ω_global(Gm)). This is equivalent to max(0, τ - H(p_m(experts)))
    loss = F.relu(threshold - current_global_entropy)
    return loss


# --- Combined Auxiliary Loss Class ---
class MoEAuxiliaryLosses(nn.Module):
    def __init__(
        self,
        num_experts: int,
        k_selected_experts: int = 1,  # K for Top-K in LoadLoss
        use_importance_loss: bool = True,
        use_load_loss: bool = True,
        use_z_loss: bool = True,
        use_local_entropy_loss: bool = True,
        use_global_entropy_loss: bool = True,
        # Params for Load Loss (from Appendix B.1 & V-MoE/Switch)
        # "σ = 1/E" where E is num_experts
        load_loss_noise_stddev_routing_factor: float = 1.0,  # Multiplier for 1/num_experts
        load_loss_noise_stddev_cdf_factor: float = 1.0,  # Multiplier for 1/num_experts
        # Global entropy thresholds (τ_text, τ_image) from Appendix A.2
        # Example: τ_text = log(9), τ_image = log(20)
        global_entropy_threshold_text: float = torch.log(torch.tensor(9.0)),
        global_entropy_threshold_image: float = torch.log(torch.tensor(20.0)),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k_selected_experts = k_selected_experts
        self.use_importance_loss = use_importance_loss
        self.use_load_loss = use_load_loss
        self.use_z_loss = use_z_loss
        self.use_local_entropy_loss = use_local_entropy_loss
        self.use_global_entropy_loss = use_global_entropy_loss

        self.noise_stddev_routing = load_loss_noise_stddev_routing_factor / num_experts
        self.noise_stddev_cdf = load_loss_noise_stddev_cdf_factor / num_experts

        self.global_entropy_threshold_text = global_entropy_threshold_text
        self.global_entropy_threshold_image = global_entropy_threshold_image
        self.eps = eps

    def forward(
        self,
        gating_probs_all: torch.Tensor,  # (total_tokens, num_experts)
        router_logits_all: torch.Tensor,  # (total_tokens, num_experts)
        modality_indices: torch.Tensor,  # (total_tokens,), 0 for image, 1 for text
        image_modality_idx: int = 0,
        text_modality_idx: int = 1,
    ) -> torch.Tensor:
        """
        Calculates the sum of enabled auxiliary losses.
        The paper mentions these are averaged and then weighted by a single coefficient (e.g., 0.04).
        This function returns the sum; the averaging and weighting should happen outside.

        Args:
            gating_probs_all: Softmax output of router for all tokens in the MoE layer.
            router_logits_all: Raw logits output of router for all tokens.
            modality_indices: Tensor indicating modality of each token.
            image_modality_idx: Integer representing image modality.
            text_modality_idx: Integer representing text modality.

        Returns:
            torch.Tensor: Sum of the enabled auxiliary losses.
        """
        total_loss = torch.tensor(
            0.0, device=gating_probs_all.device, dtype=gating_probs_all.dtype
        )
        num_active_losses = 0

        if self.use_importance_loss:
            loss_imp = importance_loss(gating_probs_all, self.num_experts, self.eps)
            total_loss += loss_imp
            num_active_losses += 1

        if self.use_load_loss:
            # Load loss is typically applied globally, not per-modality
            loss_load = load_loss(
                router_logits_all,
                self.num_experts,
                self.k_selected_experts,
                self.noise_stddev_routing,
                self.noise_stddev_cdf,
                self.eps,
            )
            total_loss += loss_load
            num_active_losses += 1

        if self.use_z_loss:
            # Z-loss is also typically global
            loss_z = z_loss_fn(router_logits_all)
            total_loss += loss_z
            num_active_losses += 1

        # Per-modality losses
        mask_image = modality_indices == image_modality_idx
        mask_text = modality_indices == text_modality_idx

        gating_probs_image = gating_probs_all[mask_image]
        gating_probs_text = gating_probs_all[mask_text]

        if self.use_local_entropy_loss:
            loss_local_ent_img = local_entropy_loss_modal(gating_probs_image, self.eps)
            loss_local_ent_txt = local_entropy_loss_modal(gating_probs_text, self.eps)
            total_loss += (
                loss_local_ent_img + loss_local_ent_txt
            )  # Summing as per paper for different modalities
            num_active_losses += 2  # Counting as two distinct losses

        if self.use_global_entropy_loss:
            loss_global_ent_img = global_entropy_loss_modal(
                gating_probs_image, self.global_entropy_threshold_image, self.eps
            )
            loss_global_ent_txt = global_entropy_loss_modal(
                gating_probs_text, self.global_entropy_threshold_text, self.eps
            )
            total_loss += loss_global_ent_img + loss_global_ent_txt
            num_active_losses += 2

        # The paper mentions: "The final aggregated auxiliary loss is computed as the average over all the losses."
        if num_active_losses > 0:
            return total_loss / num_active_losses
        else:
            return total_loss  # Should be 0.0


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_tokens_img = 100
    num_tokens_txt = 20
    total_tokens = num_tokens_img + num_tokens_txt
    n_experts = 32
    k_selected = 1  # For LoadLoss

    # --- Example Router Outputs (dummy data) ---
    # Simulate router logits (e.g., from a linear layer)
    dummy_router_logits = torch.randn(total_tokens, n_experts).to(device)
    # Simulate gating probabilities (softmax over logits)
    dummy_gating_probs = F.softmax(dummy_router_logits, dim=-1)

    # Simulate modality indices
    mod_indices = torch.cat(
        [
            torch.zeros(num_tokens_img, dtype=torch.long),
            torch.ones(num_tokens_txt, dtype=torch.long),
        ]
    ).to(device)
    img_idx, txt_idx = 0, 1

    print(f"--- Testing Individual Losses (Global) ---")
    imp_loss = importance_loss(dummy_gating_probs, n_experts)
    print(f"Importance Loss: {imp_loss.item()}")

    # For load loss, noise_stddev is usually 1.0/num_experts
    noise_std_factor = 1.0
    load_l = load_loss(
        dummy_router_logits,
        n_experts,
        k_selected,
        noise_stddev_routing=noise_std_factor / n_experts,
        noise_stddev_cdf=noise_std_factor / n_experts,
    )
    print(f"Load Loss: {load_l.item()}")

    z_l = z_loss_fn(dummy_router_logits)
    print(f"Z-Loss: {z_l.item()}")

    print(f"\n--- Testing Individual Per-Modality Losses ---")
    probs_img = dummy_gating_probs[mod_indices == img_idx]
    probs_txt = dummy_gating_probs[mod_indices == txt_idx]

    local_ent_img = local_entropy_loss_modal(probs_img)
    local_ent_txt = local_entropy_loss_modal(probs_txt)
    print(f"Local Entropy Loss (Image): {local_ent_img.item()}")
    print(f"Local Entropy Loss (Text): {local_ent_txt.item()}")

    # Thresholds τ = log(S), S = soft min experts
    # From Appendix A.2 for ablations: S_text=9, S_image=20
    thresh_img = torch.log(torch.tensor(20.0)).to(device)
    thresh_txt = torch.log(torch.tensor(9.0)).to(device)

    global_ent_img = global_entropy_loss_modal(probs_img, threshold=thresh_img)
    global_ent_txt = global_entropy_loss_modal(probs_txt, threshold=thresh_txt)
    print(f"Global Entropy Loss (Image, τ=log(20)): {global_ent_img.item()}")
    print(f"Global Entropy Loss (Text, τ=log(9)): {global_ent_txt.item()}")

    print(f"\n--- Testing MoEAuxiliaryLosses Class ---")
    aux_loss_calculator = MoEAuxiliaryLosses(
        num_experts=n_experts,
        k_selected_experts=k_selected,
        global_entropy_threshold_image=thresh_img,
        global_entropy_threshold_text=thresh_txt,
        load_loss_noise_stddev_routing_factor=noise_std_factor,
        load_loss_noise_stddev_cdf_factor=noise_std_factor,
    ).to(device)

    total_aux_loss_val = aux_loss_calculator(
        dummy_gating_probs,
        dummy_router_logits,
        mod_indices,
        image_modality_idx=img_idx,
        text_modality_idx=txt_idx,
    )
    # This is the sum of individual losses divided by the number of active losses.
    # The final coefficient (e.g., 0.04 mentioned in Appendix B.1) should be applied outside.
    print(f"Total Auxiliary Loss (averaged sum): {total_aux_loss_val.item()}")

    print(f"\n--- Testing with no text tokens (should not error) ---")
    mod_indices_no_text = torch.zeros(num_tokens_img, dtype=torch.long).to(device)
    dummy_router_logits_img_only = torch.randn(num_tokens_img, n_experts).to(device)
    dummy_gating_probs_img_only = F.softmax(dummy_router_logits_img_only, dim=-1)

    total_aux_loss_no_text = aux_loss_calculator(
        dummy_gating_probs_img_only, dummy_router_logits_img_only, mod_indices_no_text
    )
    print(f"Total Aux Loss (Image Only): {total_aux_loss_no_text.item()}")

    print(f"\n--- Testing with only local entropy for text, no global entropy ---")
    aux_loss_calculator_custom = MoEAuxiliaryLosses(
        num_experts=n_experts,
        k_selected_experts=k_selected,
        use_importance_loss=False,
        use_load_loss=False,
        use_z_loss=False,
        use_local_entropy_loss=True,  # Global local
        use_global_entropy_loss=False,  # Global global
        global_entropy_threshold_image=thresh_img,  # Will be ignored
        global_entropy_threshold_text=thresh_txt,  # Will be ignored
    ).to(device)
    # This will only compute local_ent_img and local_ent_txt and average them
    aux_loss_custom_val = aux_loss_calculator_custom(
        dummy_gating_probs, dummy_router_logits, mod_indices
    )
    expected_custom_loss = (local_ent_img + local_ent_txt) / 2.0
    print(f"Custom Aux Loss (Local Ent Only): {aux_loss_custom_val.item()}")
    print(f"Expected Custom Loss (Manual Calc): {expected_custom_loss.item()}")

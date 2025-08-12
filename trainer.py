import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.amp import autocast, GradScaler
from data.contrastive_dataloader import create_contrastive_dataloader
from model.vision.vision_encoder import MaskedVisionEncoder

try:
    import transformer_engine.pytorch as te

    HAS_TRANSFORMER_ENGINE = True
except ImportError:
    HAS_TRANSFORMER_ENGINE = False

from utils.logger_config import logger


class CustomLinearDecayLR:
    """
    Custom learning rate scheduler with linear warmup and linear decay
    as described in Sec 3.2.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        steps_per_epoch,
        max_lr=5e-4,
        min_lr=0,
        last_epoch=-1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)
        self.total_steps = int(total_epochs * steps_per_epoch)
        self.decay_steps = self.total_steps - self.warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = last_epoch + 1 if last_epoch >= 0 else 0

    def step(self):
        """Update learning rate and take a step"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1

    def get_lr(self):
        """Calculate learning rate based on current step"""
        if self.current_step < self.warmup_steps:
            # Linear warmup for 1.4 epochs up to 5e-4
            return self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Linear decay down to 0 for the remaining epochs
            decay_ratio = (self.current_step - self.warmup_steps) / self.decay_steps
            decay_factor = 1.0 - decay_ratio
            return self.max_lr * decay_factor

    def get_last_lr(self):
        """Return the last computed learning rate"""
        return [self.get_lr()]

    def state_dict(self):
        """Return the state of the scheduler"""
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        """Load the state of the scheduler"""
        self.current_step = state_dict["current_step"]


class ContrastiveTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        loss_fn="clip",  # "clip" or "siglip"
        batch_size=32,
        learning_rate=5e-4,  # Set to 5e-4 as specified
        weight_decay=1e-5,
        num_epochs=10,
        warmup_epochs=1.4,  # Warmup for 1.4 epochs
        scheduler_type="custom_linear",  # Use custom scheduler
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="checkpoints",
        save_every=1,
        local_loss=False,
        gather_with_grad=False,
        use_distributed=False,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=1,
        precision="fp32",  # "fp32", "fp16", or "fp8"
        num_workers=1,
        use_controlled_negatives=False,
        seed=42,
        use_masking=False,
        mask_ratio=0.4,
        distillation_alpha=1.0,  # α = 1
        masking_beta=2.0,  # β = 2
        create_local_crops=False,
        num_local_crops=4,
        local_crop_size=98,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_epochs
        self.scheduler_type = scheduler_type
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.use_distributed = use_distributed
        self.rank = rank
        self.world_size = world_size
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.precision = precision
        self.num_workers = num_workers
        self.use_controlled_negatives = use_controlled_negatives
        self.seed = seed
        self.use_masking = use_masking
        self.mask_ratio = mask_ratio
        self.distillation_alpha = distillation_alpha
        self.masking_beta = masking_beta
        self.warmup_epochs = warmup_epochs
        self.create_local_crops = create_local_crops
        self.num_local_crops = num_local_crops
        self.local_crop_size = local_crop_size

        # Set up mixed precision training
        self.use_amp = precision != "fp32" and device != "cpu"
        if self.use_amp:
            if precision == "fp16":
                self.scaler = GradScaler()
                logger.info("Using FP16 mixed precision training with gradient scaling")
            elif precision == "fp8":
                if (
                    HAS_TRANSFORMER_ENGINE
                    and hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()
                ):
                    self.scaler = GradScaler()
                    logger.info(
                        "Using FP8 mixed precision training with transformer engine"
                    )
                else:
                    logger.warning(
                        "FP8 precision requested but transformer_engine not available or hardware not supported. Falling back to FP16."
                    )
                    self.precision = "fp16"
                    self.scaler = GradScaler()
        else:
            self.scaler = None
            logger.info("Using FP32 precision training")

        # Create data loaders
        self.train_loader = create_contrastive_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not use_distributed,
            num_workers=self.num_workers,
            pin_memory=True,
            use_controlled_negatives=self.use_controlled_negatives,
            seed=self.seed,
            create_local_crops=self.create_local_crops,
            num_local_crops=self.num_local_crops,
            local_crop_size=self.local_crop_size,
        )

        if val_dataset:
            self.val_loader = create_contrastive_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                use_controlled_negatives=self.use_controlled_negatives,
                seed=self.seed,
                create_local_crops=self.create_local_crops,
                num_local_crops=self.num_local_crops,
                local_crop_size=self.local_crop_size,
            )
        else:
            self.val_loader = None

        # Initialize optimizer - using Adafactor as specified
        self.optimizer = optim.Adafactor(
            model.parameters(),
            lr=learning_rate,
        )

        self.loss_fn_name = loss_fn.lower()
        # Initialize loss function
        if loss_fn.lower() == "clip":
            from losses.clip_loss import ClipLoss

            self.loss_fn = ClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=True,
                rank=rank,
                world_size=world_size,
            )
        elif loss_fn.lower() == "siglip":
            from losses.sigclip_loss import SigLipLoss

            self.loss_fn = SigLipLoss(
                cache_labels=True, rank=rank, world_size=world_size
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

        # Initialize custom learning rate scheduler
        if scheduler_type == "custom_linear":
            steps_per_epoch = len(self.train_loader)
            self.scheduler = CustomLinearDecayLR(
                self.optimizer,
                warmup_epochs=self.warmup_epochs,
                total_epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                max_lr=learning_rate,
            )
        # Initialize learning rate scheduler with warmup for other types
        else:
            self.scheduler = self._get_scheduler()

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize training variables
        self.global_step = 0
        self.best_val_loss = float("inf")

        logger.info(
            f"Initialized ContrastiveTrainer with {model.__class__.__name__} on {device}"
        )

    def _get_scheduler(self):
        """Create a learning rate scheduler with warmup."""
        total_steps = self.num_epochs * len(self.train_loader)

        if self.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

    def save_checkpoint(self, epoch, val_loss=None, is_best=False):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
        )

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "val_loss": val_loss,
        }

        if self.scaler is not None:
            checkpoint_dict["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint_dict, checkpoint_path)
        logger.info(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint_dict, best_path)
            logger.info(f"Saved best model with validation loss {val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Checkpoint {checkpoint_path} does not exist. Starting from scratch."
            )
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]

        # Load scaler state if available and needed
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})"
        )
        return checkpoint["epoch"]

    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        start_time = time.time()

        # Calculate total steps for cosine momentum schedule
        total_steps = self.num_epochs * num_batches
        current_step = epoch * num_batches

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.num_epochs}",
            disable=self.rank != 0,
        )

        # Initialize optimizer gradients at the beginning of each epoch
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Calculate current step for schedules
            step = current_step + batch_idx

            # Get data from batch
            text_input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            images = batch["image"].to(self.device)

            # Get local crops if available
            local_crops = batch.get("local_crops", None)
            if local_crops is not None:
                local_crops = local_crops.to(self.device)

            # Forward pass with appropriate precision
            if self.use_amp:
                dtype = (
                    torch.float8_e4m3fn
                    if self.precision == "fp8" and HAS_TRANSFORMER_ENGINE
                    else torch.float16
                )
                with autocast(device_type=self.device, dtype=dtype):
                    # Check if we're using a masked vision encoder
                    if self.use_masking and isinstance(
                        self.model.vision_encoder, MaskedVisionEncoder
                    ):
                        # Apply masking during forward pass with local crops
                        image_features_result = self.model.vision_encoder(
                            images,
                            extract_type="cls_patch",
                            apply_masking=True,
                            step=step,
                            total_steps=total_steps,
                            local_crops=local_crops,
                        )

                        if isinstance(image_features_result, tuple):
                            # Unpack the result - features and additional info
                            image_features, mask_info = image_features_result
                            distillation_loss = mask_info.get("distillation_loss", 0.0)
                            masking_loss = mask_info.get("masking_loss", 0.0)
                            original_features = mask_info.get("original_features", None)
                        else:
                            # If no masking was applied despite the setting
                            image_features = image_features_result
                            distillation_loss = 0.0
                            masking_loss = 0.0
                            original_features = None

                        # Get text features from the model
                        text_features = self.model.text_encoder(
                            input_ids=text_input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                        )

                        # Normalize features
                        text_features = self.model.feature_extraction(text_features, "cls")
                        image_features = self.model.feature_extraction(image_features, "cls")

                        mm_images = image_features / image_features.norm(dim=1, keepdim=True)
                        mm_texts = text_features / text_features.norm(dim=1, keepdim=True)
                    else:
                        # Regular forward pass without masking
                        mm_texts, mm_images = self.model(
                            text_input_ids=text_input_ids,
                            image_features=images,
                            text_attention_mask=attention_mask,
                            text_token_type_ids=token_type_ids,
                            return_embeddings_only=True,
                        )
                        distillation_loss = 0.0
                        masking_loss = 0.0

                    # Calculate contrastive loss
                    logit_scale = self.model.logit_scale.exp()

                    if self.loss_fn_name == "clip":
                        contrastive_loss = self.loss_fn(
                            mm_images, mm_texts, logit_scale
                        )
                    elif self.loss_fn_name == "siglip":
                        logit_bias = self.model.logit_bias
                        contrastive_loss = self.loss_fn(
                            mm_images,
                            mm_texts,
                            logit_scale,
                            logit_bias=logit_bias,
                        )

                    # Combine losses with specified weights: α = 1, β = 2
                    loss = contrastive_loss
                    if distillation_loss > 0:
                        loss += (
                            self.distillation_alpha * distillation_loss
                        )  # α * distillation_loss
                    if masking_loss > 0:
                        loss += self.masking_beta * masking_loss  # β * masking_loss

                # Scale the loss by gradient accumulation steps
                loss = loss / self.gradient_accumulation_steps

                # Backward pass with scaler
                self.scaler.scale(loss).backward()

                # Update weights if we've processed enough batches for accumulation or if it's the last batch
                if (
                    (batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or batch_idx == num_batches - 1
                ):
                    # Gradient clipping with scaler
                    self.scaler.unscale_(self.optimizer)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Optimizer step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                    # Update teacher model with EMA on cosine schedule
                    if (
                        (
                            (batch_idx + 1) % self.gradient_accumulation_steps == 0
                            or batch_idx == num_batches - 1
                        )
                        and self.use_masking
                        and isinstance(self.model.vision_encoder, MaskedVisionEncoder)
                    ):
                        self.model.vision_encoder.update_teacher(
                            step=step, max_steps=total_steps
                        )

                    # Reset gradients
                    self.optimizer.zero_grad()

                    # Update global step only when optimizer step is performed
                    self.global_step += 1
            else:
                # Regular FP32 training
                # Check if we're using a masked vision encoder
                if self.use_masking and isinstance(
                    self.model.vision_encoder, MaskedVisionEncoder
                ):
                    # Apply masking during forward pass
                    image_features_result = self.model.vision_encoder(
                        images,
                        extract_type="cls_patch",
                        apply_masking=True,
                        step=step,
                        total_steps=total_steps,
                        local_crops=local_crops,
                    )

                    if isinstance(image_features_result, tuple):
                        # Unpack the result - features and additional info
                        image_features, mask_info = image_features_result
                        distillation_loss = mask_info.get("distillation_loss", 0.0)
                        masking_loss = mask_info.get("masking_loss", 0.0)
                        original_features = mask_info.get("original_features", None)
                    else:
                        # If no masking was applied despite the setting
                        image_features = image_features_result
                        distillation_loss = 0.0
                        masking_loss = 0.0
                        original_features = None

                    # Get text features from the model
                    text_features = self.model.text_encoder(
                        input_ids=text_input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    # Regular forward pass without masking
                    text_features, image_features = self.model(
                        text_input_ids=text_input_ids,
                        image_features=images,
                        text_attention_mask=attention_mask,
                        text_token_type_ids=token_type_ids,
                    )
                    distillation_loss = 0.0
                    masking_loss = 0.0

                # Calculate contrastive loss
                logit_scale = self.model.logit_scale.exp()

                if self.loss_fn_name == "clip":
                    contrastive_loss = self.loss_fn(
                        image_features, text_features, logit_scale
                    )
                elif self.loss_fn_name == "siglip":
                    logit_bias = self.model.logit_bias
                    contrastive_loss = self.loss_fn(
                        image_features,
                        text_features,
                        logit_scale,
                        logit_bias=logit_bias,
                    )

                # Combine losses: contrastive loss + distillation loss
                loss = contrastive_loss
                if distillation_loss > 0:
                    loss += (
                        self.distillation_alpha * distillation_loss
                    )  # α * distillation_loss
                if masking_loss > 0:
                    loss += self.masking_beta * masking_loss  # β * masking_loss

                # Scale the loss by gradient accumulation steps
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights if we've processed enough batches for accumulation or if it's the last batch
                if (
                    (batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or batch_idx == num_batches - 1
                ):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()

                    # Update teacher model if using self-distillation
                    if (
                        self.use_masking
                        and isinstance(self.model.vision_encoder, MaskedVisionEncoder)
                        and hasattr(self.model.vision_encoder, "update_teacher")
                    ):
                        self.model.vision_encoder.update_teacher()

                    # Reset gradients
                    self.optimizer.zero_grad()

                    # Update global step only when optimizer step is performed
                    self.global_step += 1

            # For logging, use the unscaled loss value
            batch_loss = loss.item() * self.gradient_accumulation_steps
            epoch_loss += batch_loss

            # Update progress bar with additional info
            if self.use_masking and isinstance(
                self.model.vision_encoder, MaskedVisionEncoder
            ):
                progress_bar.set_postfix(
                    {
                        "total_loss, cont_loss, dist_loss, mask_loss, momentum": f"{batch_loss:.4f}, {(self.distillation_alpha * distillation_loss.item()):.4f}, {(self.masking_beta * masking_loss.item()):.4f}, {self.model.vision_encoder.current_momentum.item():.6f}",
                    }
                )
            else:
                progress_bar.set_postfix(
                    {
                        "train_loss": batch_loss,
                        "lr": self.scheduler.get_last_lr()[0],
                        "precision": self.precision,
                    }
                )

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time

        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s - avg train loss: {avg_epoch_loss:.4f}"
        )

        return avg_epoch_loss

    def evaluate(self):
        """Evaluate the model on the validation set."""
        if not self.val_loader:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader, desc="Evaluating", disable=self.rank != 0
            ):
                # Get data from batch
                text_input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                images = batch["image"].to(self.device)

                # Forward pass with appropriate precision for evaluation
                if self.use_amp:
                    dtype = (
                        torch.float8_e4m3fn
                        if self.precision == "fp8" and HAS_TRANSFORMER_ENGINE
                        else torch.float16
                    )
                    with autocast(device_type=self.device, dtype=dtype):
                        text_features, image_features = self.model(
                            text_input_ids=text_input_ids,
                            image_features=images,
                            text_attention_mask=attention_mask,
                            text_token_type_ids=token_type_ids,
                        )

                        # Calculate loss
                        logit_scale = self.model.logit_scale.exp()
                        if self.loss_fn_name == "clip":
                            loss = self.loss_fn(
                                image_features, text_features, logit_scale
                            )
                        elif self.loss_fn_name == "siglip":
                            logit_bias = self.model.logit_bias
                            loss = self.loss_fn(
                                image_features,
                                text_features,
                                logit_scale,
                                logit_bias=logit_bias,
                            )
                else:
                    # Regular FP32 evaluation
                    text_features, image_features = self.model(
                        text_input_ids=text_input_ids,
                        image_features=images,
                        text_attention_mask=attention_mask,
                        text_token_type_ids=token_type_ids,
                    )

                    # Calculate loss
                    logit_scale = self.model.logit_scale.exp()
                    if self.loss_fn_name == "clip":
                        loss = self.loss_fn(image_features, text_features, logit_scale)
                    elif self.loss_fn_name == "siglip":
                        logit_bias = self.model.logit_bias
                        loss = self.loss_fn(
                            image_features,
                            text_features,
                            logit_scale,
                            logit_bias=logit_bias,
                        )

                total_loss += loss.item()

        avg_val_loss = total_loss / num_batches
        logger.info(f"Validation loss: {avg_val_loss:.4f}")

        return avg_val_loss

    def train(self, resume_from=None):
        """Train the model for the specified number of epochs."""
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1

        logger.info(f"Starting training from epoch {start_epoch}")

        for epoch in range(start_epoch, self.num_epochs):
            train_loss = self.train_epoch(epoch)

            # Evaluate on validation set
            val_loss = self.evaluate() if self.val_loader else None

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or epoch == self.num_epochs - 1:
                is_best = False
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best = True
                self.save_checkpoint(epoch, val_loss, is_best)

        logger.info("Training completed!")
        logger.info(
            f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.global_step // len(self.train_loader)}"
        )

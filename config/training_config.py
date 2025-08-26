from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Dataset parameters
    dataset_name: str = "nhq188/moe-dataset-2"
    image_column: str = "image"
    caption_column: str = "caption"

    # Training parameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Model parameters
    loss_fn: str = "clip"  # "clip" or "siglip"

    # Two-stage training
    training_stage: str = "contrastive"  # "teacher" or "contrastive"
    
    # Teacher Learning Stage parameters
    parallel_text_dataset: str = ""  # Path to parallel text corpus
    teacher_epochs: int = 5  # Number of epochs for teacher learning stage
    
    # Contrastive Learning Stage parameters  
    contrastive_epochs: int = 10  # Number of epochs for contrastive learning stage
    multilingual_data_ratio: float = 1.0  # Ratio of multilingual to bilingual data

    # Logging and checkpointing
    output_dir: str = "outputs"
    save_every: int = 1
    log_every: int = 100

    # Distributed training
    use_distributed: bool = False

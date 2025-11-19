"""
Example script demonstrating how to use ImageTextMultiModalForCausalLM
for text generation with greedy search and beam search
"""

import torch
from transformers import AutoTokenizer
import clip


# Try different import paths for BeamSearchScorer (concrete implementation, not abstract BeamScorer)
BeamSearchScorer = None
try:
    from transformers.generation_beam_search import BeamSearchScorer
except ImportError:
    try:
        from transformers import BeamSearchScorer
    except ImportError:
        try:
            from transformers.generation.beam_search import BeamSearchScorer
        except ImportError:
            raise ImportError(
                "Could not import BeamSearchScorer. Please check your transformers version.\n"
                "You may need to update transformers: pip install --upgrade transformers\n"
                "Or install a compatible version: pip install transformers>=4.20.0"
            )

# Try different import paths for stopping criteria depending on transformers version
try:
    from transformers.generation_stopping_criteria import (
        MaxLengthCriteria,
        StoppingCriteriaList,
    )
except ImportError:
    try:
        # For newer transformers versions
        from transformers import MaxLengthCriteria, StoppingCriteriaList
    except ImportError:
        try:
            # Alternative import path
            from transformers.generation.stopping_criteria import (
                MaxLengthCriteria,
                StoppingCriteriaList,
            )
        except ImportError:
            # Fallback: create simple implementations
            class MaxLengthCriteria:
                def __init__(self, max_length):
                    self.max_length = max_length

                def __call__(self, input_ids, scores):
                    return input_ids.shape[-1] >= self.max_length

            class StoppingCriteriaList:
                def __init__(self, criteria=None):
                    self.criteria = criteria or []
                    # Store max_length from MaxLengthCriteria if present
                    self.max_length = None
                    for c in self.criteria:
                        if isinstance(c, MaxLengthCriteria):
                            self.max_length = c.max_length
                            break

                def __call__(self, input_ids, scores):
                    return any(c(input_ids, scores) for c in self.criteria)


from model.mm.mm_encoder_vl import ImageTextMultiModalEncoder
from model.mm.mm_encoder_vl_causal import ImageTextMultiModalForCausalLM
from model.builder import build_vision_encoder, build_text_encoder
from model.config import MultiModalConfig


def load_model(checkpoint_path=None):
    """Load the model"""
    # Build encoders
    vision_encoder = build_vision_encoder()
    text_encoder = build_text_encoder()

    # Create config
    config = MultiModalConfig()
    config.image_size = [224, 224]
    config.patch_size = [32, 32]

    # Create base model
    base_model = ImageTextMultiModalEncoder(
        text_encoder=text_encoder, vision_encoder=vision_encoder, config=config
    )

    # Load weights if checkpoint provided
    if checkpoint_path:
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            base_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            base_model.load_state_dict(checkpoint, strict=False)
        print("Weights loaded!")

    # Create causal LM model
    model = ImageTextMultiModalForCausalLM(
        base_model=base_model, config=base_model.roberta_config
    )

    return model


def generate_greedy(
    model,
    tokenizer,
    image_input,
    text_input_ids,
    max_length=50,
    pad_token_id=1,
    eos_token_id=2,
):
    """
    Generate text using greedy search

    Args:
        model: ImageTextMultiModalForCausalLM model
        tokenizer: Tokenizer
        image_input: Image tensor [batch_size, 3, H, W]
        text_input_ids: Initial text input ids [batch_size, seq_len]
        max_length: Maximum generation length
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID

    Returns:
        Generated text
    """
    model.eval()

    with torch.no_grad():
        # Generate using greedy search
        generated_ids = model.greedy_search(
            input_ids=text_input_ids,
            image_input=image_input,  # Passed via model_kwargs
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=False,
        )

    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def generate_beam_search(
    model,
    tokenizer,
    image_input,
    text_input_ids,
    num_beams=5,
    max_length=50,
    pad_token_id=1,
    eos_token_id=2,
):
    """
    Generate text using beam search

    Args:
        model: ImageTextMultiModalForCausalLM model
        tokenizer: Tokenizer
        image_input: Image tensor [batch_size, 3, H, W]
        text_input_ids: Initial text input ids [batch_size, seq_len]
        num_beams: Number of beams
        max_length: Maximum generation length
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID

    Returns:
        Generated text
    """
    model.eval()

    # Create beam scorer using BeamSearchScorer (concrete implementation)
    if BeamSearchScorer is None:
        raise ValueError(
            "BeamSearchScorer is not available. Please install transformers>=4.20.0: "
            "pip install --upgrade transformers"
        )

    # Create beam scorer with proper arguments
    beam_scorer = BeamSearchScorer(
        batch_size=text_input_ids.shape[0],
        num_beams=num_beams,
        device=text_input_ids.device,
        length_penalty=1.0,
        do_early_stopping=False,
    )

    # Create stopping criteria
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

    # Expand input_ids and image_input for beam search
    # Each input needs to be replicated num_beams times
    batch_size = text_input_ids.shape[0]
    expanded_input_ids = text_input_ids.repeat_interleave(num_beams, dim=0)
    expanded_image_input = image_input.repeat_interleave(num_beams, dim=0)

    with torch.no_grad():
        # Generate using beam search
        generated_ids = model.beam_search(
            input_ids=expanded_input_ids,
            beam_scorer=beam_scorer,
            image_input=expanded_image_input,  # Passed via model_kwargs
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=False,
        )

    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from image")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--image_path", type=str, help="Path to image file")
    parser.add_argument(
        "--method",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="Generation method",
    )
    parser.add_argument(
        "--num_beams", type=int, default=5, help="Number of beams for beam search"
    )
    parser.add_argument(
        "--max_length", type=int, default=50, help="Maximum generation length"
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Text model name for tokenizer",
    )
    # /Users/gos/Documents/Moe/moe/uit-vilc/uitvic_dataset/coco_uitvic_test/coco_uitvic_test/000000363257.jpg
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Model loaded!")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    print("Tokenizer loaded!")

    # Prepare image (you'll need to load and preprocess your image)
    # This is just an example - replace with actual image loading
    print("Preparing image...")

    from PIL import Image

    image = Image.open(args.image_path).convert("RGB")
    _, transform = clip.load("ViT-B/32", device="cpu")
    image_input = transform(image).unsqueeze(0).to(device)

    # Prepare initial text (e.g., start token or prompt)
    print("Preparing text input...")
    initial_text = "<s> Ảnh đề cập về:"  # Start token
    text_input_ids = tokenizer.encode(initial_text, return_tensors="pt").to(device)

    # Generate
    print(f"\nGenerating text using {args.method} search...")
    if args.method == "greedy":
        generated_text = generate_greedy(
            model=model,
            tokenizer=tokenizer,
            image_input=image_input,
            text_input_ids=text_input_ids,
            max_length=args.max_length,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else 1,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 2,
        )
    else:  # beam search
        generated_text = generate_beam_search(
            model=model,
            tokenizer=tokenizer,
            image_input=image_input,
            text_input_ids=text_input_ids,
            num_beams=args.num_beams,
            max_length=args.max_length,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else 1,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 2,
        )

    print(f"\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main()

"""
ImageTextMultiModalForCausalLM - Vision-Language Model with Decoder for Generation
Supports greedy search and beam search for text generation
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List, Tuple, Union
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# Try different import paths for generation utilities depending on transformers version
try:
    from transformers.generation_logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
    )
except ImportError:
    try:
        from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
    except ImportError:
        try:
            from transformers.generation.logits_process import (
                LogitsProcessorList,
                RepetitionPenaltyLogitsProcessor,
            )
        except ImportError:
            # Fallback: simple implementation
            class LogitsProcessorList(list):
                def __call__(self, input_ids, scores):
                    for processor in self:
                        scores = processor(input_ids, scores)
                    return scores

            # Fallback: simple RepetitionPenaltyLogitsProcessor
            class RepetitionPenaltyLogitsProcessor:
                def __init__(self, penalty: float = 1.0):
                    if not isinstance(penalty, float) or not (penalty > 0):
                        raise ValueError(
                            f"`penalty` has to be a strictly positive float, but is {penalty}"
                        )
                    self.penalty = penalty

                def __call__(
                    self, input_ids: torch.LongTensor, scores: torch.FloatTensor
                ) -> torch.FloatTensor:
                    score = torch.gather(scores, 1, input_ids)
                    # if score < 0 then repetition penalty has to be multiplied to reduce the token probability
                    score = torch.where(
                        score < 0, score * self.penalty, score / self.penalty
                    )
                    scores.scatter_(1, input_ids, score)
                    return scores


try:
    from transformers.generation_stopping_criteria import (
        StoppingCriteriaList,
        validate_stopping_criteria,
    )
except ImportError:
    try:
        from transformers import StoppingCriteriaList, validate_stopping_criteria
    except ImportError:
        try:
            from transformers.generation.stopping_criteria import (
                StoppingCriteriaList,
                validate_stopping_criteria,
            )
        except ImportError:
            # Fallback implementations
            class MaxLengthCriteria:
                def __init__(self, max_length):
                    self.max_length = max_length

                def __call__(self, input_ids, scores):
                    return input_ids.shape[-1] >= self.max_length

            class StoppingCriteriaList:
                def __init__(self, criteria=None):
                    self.criteria = criteria or []
                    self.max_length = None
                    for c in self.criteria:
                        if isinstance(c, MaxLengthCriteria):
                            self.max_length = c.max_length
                            break

                def __call__(self, input_ids, scores):
                    return any(c(input_ids, scores) for c in self.criteria)

            def validate_stopping_criteria(stopping_criteria, max_length):
                if stopping_criteria is None:
                    stopping_criteria = StoppingCriteriaList()
                if max_length is not None:
                    # Use the MaxLengthCriteria we defined above
                    stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
                return stopping_criteria


try:
    from transformers.generation_utils import (
        GreedySearchOutput,
        GreedySearchDecoderOnlyOutput,
        BeamSearchOutput,
        BeamSearchDecoderOnlyOutput,
    )
except ImportError:
    try:
        from transformers import (
            GreedySearchOutput,
            GreedySearchDecoderOnlyOutput,
            BeamSearchOutput,
            BeamSearchDecoderOnlyOutput,
        )
    except ImportError:
        # Fallback: simple dataclasses
        from dataclasses import dataclass
        from typing import Optional, Tuple

        @dataclass
        class GreedySearchDecoderOnlyOutput:
            sequences: torch.Tensor
            scores: Optional[Tuple[torch.Tensor]] = None
            attentions: Optional[Tuple[Tuple[torch.Tensor]]] = None
            hidden_states: Optional[Tuple[Tuple[torch.Tensor]]] = None

        @dataclass
        class BeamSearchDecoderOnlyOutput:
            sequences: torch.Tensor
            sequences_scores: Optional[torch.Tensor] = None
            scores: Optional[Tuple[torch.Tensor]] = None
            beam_indices: Optional[torch.Tensor] = None
            attentions: Optional[Tuple[Tuple[torch.Tensor]]] = None
            hidden_states: Optional[Tuple[Tuple[torch.Tensor]]] = None

        GreedySearchOutput = GreedySearchDecoderOnlyOutput
        BeamSearchOutput = BeamSearchDecoderOnlyOutput

try:
    from transformers.generation_beam_search import BeamScorer
except ImportError:
    try:
        from transformers import BeamScorer
    except ImportError:
        try:
            from transformers.generation.beam_search import BeamScorer
        except ImportError:
            # This will cause an error if actually used, but at least allows import
            BeamScorer = None

try:
    from transformers.pytorch_utils import torch_int_div
except ImportError:
    try:
        from transformers.utils import torch_int_div
    except ImportError:
        # Fallback implementation
        def torch_int_div(tensor, other):
            """Integer division for tensors"""
            return tensor // other


from model.mm.mm_encoder_vl import ImageTextMultiModalEncoder


class ImageTextMultiModalForCausalLM(nn.Module):
    """
    Vision-Language Causal LM for text generation
    Similar to ImageTextForCausalLM but based on ImageTextMultiModalEncoder
    """

    def __init__(self, base_model: ImageTextMultiModalEncoder, config=None):
        super(ImageTextMultiModalForCausalLM, self).__init__()

        self.base_model = base_model
        self.config = config if config is not None else base_model.roberta_config

        # Set is_decoder to True for generation
        if hasattr(self.config, "is_decoder"):
            if not self.config.is_decoder:
                warnings.warn(
                    "Setting config.is_decoder=True for generation. "
                    "If you want to use this model for generation, make sure is_decoder=True."
                )
                self.config.is_decoder = True

        # LM head for generation
        self.lm_head = RobertaLMHead(self.config)

        # Image input cache for generation
        self.__image_input_cache__ = None

    @property
    def get_num_patches(self):
        """Get number of image patches"""
        return self.base_model.get_num_patches()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Legacy parameters
        text_input_ids: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for causal LM

        Args:
            input_ids: Text input ids
            image_input: Image input tensor
            labels: Labels for training (optional)
            past_key_values: Past key values for generation
            ... other parameters

        Returns:
            CausalLMOutputWithCrossAttentions
        """
        # Handle legacy API
        if text_input_ids is not None:
            input_ids = text_input_ids
        if image_features is not None:
            image_input = image_features
        if text_attention_mask is not None:
            attention_mask = text_attention_mask
        if text_token_type_ids is not None:
            token_type_ids = text_token_type_ids

        return_dict = return_dict if return_dict is not None else True
        if labels is not None:
            use_cache = False

        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            image_input=image_input,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]

        # Get prediction scores from LM head
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # Get text sequence length (image embeddings are concatenated after text)
            text_seq_len = input_ids.shape[1]

            # Only get prediction scores for text tokens (exclude image embeddings)
            prediction_scores_text = prediction_scores[:, :text_seq_len, :]

            # Shift prediction scores and labels for next-token prediction
            shifted_prediction_scores = prediction_scores_text[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=(
                outputs.past_key_values if hasattr(outputs, "past_key_values") else None
            ),
            hidden_states=(
                outputs.hidden_states if hasattr(outputs, "hidden_states") else None
            ),
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, past=None, **model_kwargs
    ):
        """
        Prepare inputs for generation
        Similar to ImageTextForCausalLM.prepare_inputs_for_generation
        """
        if past is not None:
            # Handle inputs with past (for incremental generation)
            assert self.__image_input_cache__ is not None, (
                "self.__image_input_cache__ is None. "
                "Check `image_input` in your `inputs`, "
                "Or something wrong with this function when being called for first time."
            )

            batch_length = input_ids.shape[0]
            num_patches = self.get_num_patches

            last_input_ids = input_ids[:, -1:]
            attention_mask = torch.ones_like(input_ids)
            last_token_type_ids = torch.zeros_like(last_input_ids)

            # Extend attention mask to cover image patches
            extra_attention_mask = torch.ones(
                batch_length,
                num_patches,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((attention_mask, extra_attention_mask), dim=1)

            # Remove image modality from past_key_values
            # Because past is (text, image) and when Transformer appends new token,
            # new token is appended to past, making it (text, image, new token)
            # which breaks the sequence. So we need to remove image from past.
            new_past = tuple(
                (
                    tuple((past_key[:, :, :-num_patches, :] for past_key in layer))
                    for layer in past
                )
            )
            return {
                "input_ids": last_input_ids,
                "past_key_values": new_past,
                "attention_mask": attention_mask,
                "token_type_ids": last_token_type_ids,
                "image_input": self.__image_input_cache__,
            }
        else:
            # Handle very first inputs
            # Handle beam search case
            image_input = model_kwargs.get("image_input", None)
            if image_input is not None:
                # Only repeat if batch sizes don't match
                # In beam search, image_input should already be expanded in example_generation.py
                # So we only need to repeat if image_input has batch_size=1 and input_ids has larger batch
                if image_input.shape[0] == 1 and input_ids.shape[0] > 1:
                    # Repeat image_input to match input_ids batch size
                    self.__image_input_cache__ = image_input.repeat(
                        input_ids.shape[0], 1, 1, 1
                    )
                elif image_input.shape[0] == input_ids.shape[0]:
                    # Already matched, use as is
                    self.__image_input_cache__ = image_input
                else:
                    # Unexpected case: batch sizes don't match and neither is 1
                    raise ValueError(
                        f"Batch size mismatch: image_input has batch_size={image_input.shape[0]}, "
                        f"but input_ids has batch_size={input_ids.shape[0]}. "
                        f"Please ensure image_input is expanded to match input_ids batch size."
                    )
            return {
                "input_ids": input_ids,
                "attention_mask": model_kwargs.get("attention_mask", None),
                "token_type_ids": model_kwargs.get("token_type_ids", None),
                "image_input": self.__image_input_cache__,
            }

    def _reorder_cache(self, past, beam_idx):
        """
        Reorder cache for beam search
        Copied from RoBERTaForCausalLM
        """
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False
    ):
        """Update model kwargs for generation"""
        # Update past_key_values
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            model_kwargs["past"] = outputs.past_key_values
        elif "past" not in model_kwargs:
            model_kwargs["past"] = None

        # Update attention mask - extend to cover new token
        if (
            "attention_mask" in model_kwargs
            and model_kwargs["attention_mask"] is not None
        ):
            attention_mask = model_kwargs["attention_mask"]
            # Get num_patches to extend mask correctly
            num_patches = self.get_num_patches
            # Extend attention mask: [batch, seq_len] -> [batch, seq_len + 1]
            # Also need to account for image patches
            new_attention = attention_mask.new_ones((attention_mask.shape[0], 1))
            image_attention = attention_mask.new_ones(
                (attention_mask.shape[0], num_patches)
            )
            # If attention_mask already includes image patches, just extend text part
            if attention_mask.shape[1] > num_patches:
                # Already has image patches, just extend text
                text_mask = attention_mask[:, :-num_patches]
                image_mask = attention_mask[:, -num_patches:]
                model_kwargs["attention_mask"] = torch.cat(
                    [torch.cat([text_mask, new_attention], dim=-1), image_mask], dim=-1
                )
            else:
                # Doesn't have image patches yet, add both
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        torch.cat([attention_mask, new_attention], dim=-1),
                        image_attention,
                    ],
                    dim=-1,
                )

        return model_kwargs

    def adjust_logits_during_generation(self, logits, cur_len=None):
        """Adjust logits during generation (can be overridden)"""
        return logits

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        repetition_penalty: Optional[float] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        """
        Greedy search generation
        Copied from transformers.generation_utils.GenerationMixin.greedy_search
        """
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )

        # Add repetition penalty if specified
        if repetition_penalty is not None and repetition_penalty != 1.0:
            logits_processor.append(
                RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
            )

        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else getattr(self.config, "pad_token_id", 1)
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.config, "eos_token_id", 2)
        )
        output_scores = output_scores if output_scores is not None else False
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else False
        )

        # Init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # Keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                if this_peer_finished_flag.item() == 0.0:
                    break

            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # Forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # Get the last token of outputs with respect to text modality
            # Need to account for image patches in sequence
            next_token_logits = outputs.logits[:, -(1 + self.get_num_patches), :]

            # Pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # Argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # Finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # Update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )
            cur_len = cur_len + 1

            # If eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )

            # Stop when each sentence is finished, or if we exceed the maximum length
            stop_generation = False
            if unfinished_sequences.max() == 0:
                stop_generation = True
            else:
                stop_result = stopping_criteria(input_ids, scores)
                # Handle both boolean and tensor returns
                if isinstance(stop_result, torch.Tensor):
                    stop_generation = (
                        stop_result.any().item() if stop_result.numel() > 0 else False
                    )
                else:
                    stop_generation = bool(stop_result)

            if stop_generation:
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        repetition_penalty: Optional[float] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        """
        Beam search generation
        Copied from transformers.generation_utils.GenerationMixin.beam_search
        """
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )

        # Add repetition penalty if specified
        if repetition_penalty is not None and repetition_penalty != 1.0:
            logits_processor.append(
                RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
            )

        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning,
            )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else getattr(self.config, "pad_token_id", 1)
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.config, "eos_token_id", 2)
        )
        output_scores = output_scores if output_scores is not None else False
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else False
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # Init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size))
            if (return_dict_in_generate and output_scores)
            else None
        )
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue

            # Get the last token of outputs with respect to text modality
            next_token_logits = outputs.logits[:, -(1 + self.get_num_patches), :]

            # Adjust logits
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len
            )
            next_token_scores = F.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[
                :, None
            ].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # Reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # Stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(
                    model_kwargs["past"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple(
                    (
                        beam_indices[beam_idx[i]] + (beam_idx[i],)
                        for i in range(len(beam_indices))
                    )
                )

            # Increase cur_len
            cur_len = cur_len + 1

            # Check stopping criteria - handle both boolean and tensor returns
            stop_generation = False
            if beam_scorer.is_done:
                stop_generation = True
            else:
                stop_result = stopping_criteria(input_ids, scores)
                # Handle both boolean and tensor returns
                if isinstance(stop_result, torch.Tensor):
                    stop_generation = (
                        stop_result.any().item() if stop_result.numel() > 0 else False
                    )
                else:
                    stop_generation = bool(stop_result)

            if stop_generation:
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return sequence_outputs["sequences"]


import torch
from transformers import PreTrainedModel

def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
    ) -> dict:
    # TODO 1: get the log probabilities of the response tokens in `labels`
    # using the provided `model` and `input_ids`.
    
    model_outputs = model(input_ids=input_ids)
    logits = model_outputs.logits

    log_probs_all = torch.log_softmax(logits, dim=-1) 
    log_probs = torch.gather(log_probs_all, dim=2, index=labels.unsqueeze(-1))
    log_probs = log_probs.squeeze(-1)  # Shape: (batch_size, seq_len)
    token_entropy = compute_entropy(logits) if return_token_entropy else None

    return {"log_probs": log_probs, "token_entropy": token_entropy}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of the probability distribution defined by the logits.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)

    Returns:
        Tensor of shape (batch_size, seq_len) representing the entropy at each position.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    Tokenize prompts and outputs, creating proper masks for response tokens.

    Args:
        prompt_strs: List of prompt strings
        output_strs: List of output/response strings
        tokenizer: Tokenizer instance

    Returns:
        dict with input_ids, labels, and response_mask tensors
    """
    all_input_ids = []
    all_labels = []
    all_response_masks = []
    prompt_lens = []

    # First pass: tokenize everything to find the max length
    combined_tokenized = []
    for prompt, output in zip(prompt_strs, output_strs):
        # Tokenize prompt and output separately then concatenate
        # This preserves the correct token boundaries
        prompt_ids = tokenizer(prompt, add_special_tokens=True)['input_ids']
        output_ids = tokenizer(output, add_special_tokens=True)['input_ids']

        prompt_lens.append(len(prompt_ids))

        # Concatenate the token IDs
        combined_ids = prompt_ids + output_ids
        combined_tokenized.append(combined_ids)

    # Find max length of combined sequences
    max_combined_len = max(len(ids) for ids in combined_tokenized)

    # We need to pad to at least max_combined_len + 1 so after slicing we get the right size
    # The output size should be max_combined_len (not max_combined_len - 1)
    padded_len = max_combined_len

    # Second pass: create padded tensors
    for i, (combined_ids, prompt_len) in enumerate(zip(combined_tokenized, prompt_lens)):
        full_len = len(combined_ids)

        # Create padded tensor - pad to padded_len to ensure we have enough after slicing
        padded_ids = torch.full((padded_len,), tokenizer.pad_token_id, dtype=torch.long)
        padded_ids[:full_len] = torch.tensor(combined_ids, dtype=torch.long)

        # Slice off last token for input_ids
        input_ids = padded_ids[:-1]  # Shape: (max_combined_len,)

        # Labels are shifted by 1
        labels = padded_ids[1:]  # Shape: (max_combined_len,)

        # Create response mask:
        # - 0 for prompt tokens
        # - 0 for padding tokens
        # - 1 for actual response tokens
        response_mask = torch.zeros(max_combined_len-1, dtype=torch.float32)

        # Response starts at prompt_len-1 (due to shift) and goes until actual content ends
        response_start = prompt_len - 1
        response_end = min(full_len - 1, max_combined_len-1)  # -1 because of shift

        if response_start < response_end:
            response_mask[response_start:response_end] = 1.0

        all_input_ids.append(input_ids.unsqueeze(0))
        all_labels.append(labels.unsqueeze(0))
        all_response_masks.append(response_mask.unsqueeze(0))

    # Stack all batches
    return {
        'input_ids': torch.cat(all_input_ids, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'response_mask': torch.cat(all_response_masks, dim=0)
    } 
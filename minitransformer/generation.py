import torch
import torch.nn.functional as F
from .model import TransformerLM


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_uncached(
    model: TransformerLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> torch.Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    curr_tokens = prompt_tokens.clone()
    if max_new_tokens == 0:
        return curr_tokens
    for _ in range(max_new_tokens):
        logits, _ = model(curr_tokens, use_cache=False)
        next_token_logits = logits[:, -1, :]

        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)

        curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)
    return curr_tokens


@torch.no_grad()
def generate_cached(
    model: TransformerLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> torch.Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    curr_tokens = prompt_tokens.clone()
    if max_new_tokens == 0:
        return curr_tokens
    batch_size, prompt_len = prompt_tokens.shape

    logits, past_key_values = model(prompt_tokens, use_cache=True, start_pos=0)
    next_token_logits = logits[:, -1, :]

    if temperature == 0.0:
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    else:
        probs = F.softmax(next_token_logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)

    curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)
    current_pos = prompt_len

    for _ in range(max_new_tokens - 1):
        logits, past_key_values = model(
            next_token, past_key_values=past_key_values, use_cache=True, start_pos=current_pos
        )
        next_token_logits = logits[:, -1, :]

        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)

        curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)
        current_pos += 1

    return curr_tokens

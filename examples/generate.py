"""Run a tiny randomly initialized model on CPU or CUDA."""
import torch
from minitransformer import ModelConfig, TransformerLM, generate_cached


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    )
    model = TransformerLM(config).to(device).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
    with torch.inference_mode():
        output = generate_cached(model, prompt, max_new_tokens=12)
    print(f"device={device}, token_ids={output[0].tolist()}")


if __name__ == "__main__":
    main()

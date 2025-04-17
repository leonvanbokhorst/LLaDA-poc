#!/usr/bin/env python3
"""
Generate text using the trained masked-diffusion language model.
"""
import argparse
import torch
import random
from transformers import BertTokenizerFast
from model import DiffusionConfig, DiffusionTransformer


def generate_text(
    model: DiffusionTransformer,
    tokenizer: BertTokenizerFast,
    prompt: str,
    max_gen_len: int,
    steps: int,
    remask_ratio: float,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> str:
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    seq_len = len(prompt_ids) + max_gen_len
    # Initialize sequence: prompt + masks
    mask_id = tokenizer.mask_token_id
    sequence = prompt_ids + [mask_id] * max_gen_len
    sequence = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for step in range(steps):
            # get model predictions
            logits = model(
                sequence, attention_mask=(sequence != tokenizer.pad_token_id).long()
            )
            # Fill all current masks with probabilistic sampling
            mask_positions = sequence == mask_id
            for idx in mask_positions.nonzero(as_tuple=False):
                b, pos = idx.tolist()
                raw_logits = logits[b, pos]  # [vocab_size]
                # apply temperature
                scaled_logits = raw_logits / temperature
                # top-k filtering
                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(scaled_logits, top_k)
                    mask_vals = torch.full_like(scaled_logits, float("-inf"))
                    mask_vals[topk_idx] = scaled_logits[topk_idx]
                    filtered_logits = mask_vals
                else:
                    filtered_logits = scaled_logits
                # top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(
                        filtered_logits, descending=True
                    )
                    cum_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    # determine cutoff index
                    cutoff_mask = cum_probs > top_p
                    if cutoff_mask.any():
                        cutoff_idx = torch.nonzero(cutoff_mask, as_tuple=False)[
                            0
                        ].item()
                        remove_indices = sorted_idx[cutoff_idx + 1 :]
                        filtered_logits[remove_indices] = float("-inf")
                # sample next token
                token_probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(token_probs, num_samples=1).item()
                sequence[b, pos] = next_token
            # Remask for next iteration
            if step < steps - 1:
                mask_positions = sequence != mask_id
                for idx in mask_positions.nonzero(as_tuple=False):
                    b, pos = idx.tolist()
                    if pos >= len(prompt_ids) and random.random() < remask_ratio:
                        sequence[b, pos] = mask_id
    # Decode and return
    output_ids = sequence[0].tolist()
    # Remove pad and mask tokens
    cleaned = [i for i in output_ids if i != mask_id and i != tokenizer.pad_token_id]
    return tokenizer.decode(cleaned, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with masked-diffusion model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="Max sequence length model was trained with",
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Prompt text to condition on"
    )
    parser.add_argument(
        "--max_gen_len", type=int, default=50, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--remask_ratio",
        type=float,
        default=0.5,
        help="Fraction of tokens to remask each step",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=0, help="Top-K sampling (0=disabled)"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Initialize tokenizer and model config
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    config = DiffusionConfig(
        vocab_size=len(tokenizer),
        mask_token_id=tokenizer.mask_token_id,
        max_position_embeddings=args.max_seq_len,
    )
    model = DiffusionTransformer(config).to(device)
    # Load checkpoint
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_gen_len=args.max_gen_len,
        steps=args.steps,
        remask_ratio=args.remask_ratio,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()

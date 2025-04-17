#!/usr/bin/env python3
"""
Train a masked-diffusion language model (LLaDA POC).
"""
from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizerFast
from tqdm.auto import tqdm

from model import DiffusionConfig, DiffusionTransformer


def mask_inputs(
    input_ids: torch.Tensor, mask_token_id: int, epsilon: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly mask tokens according to diffusion time t for each sample.
    Returns (noisy_input_ids, mask_positions, p_mask).
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    # Sample diffusion time uniformly, mask with ninja-like randomness
    t = torch.rand((batch_size, 1), device=device)
    p_mask = t.expand(-1, seq_len)
    rand = torch.rand((batch_size, seq_len), device=device)
    mask_positions = rand < p_mask
    noisy_input_ids = input_ids.clone()
    noisy_input_ids[mask_positions] = mask_token_id
    return noisy_input_ids, mask_positions, p_mask


def compute_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask_positions: torch.Tensor,
    p_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute weighted cross-entropy loss on masked positions.
    """
    batch_size, seq_len, vocab_size = logits.shape
    epsilon = 1e-5
    logits_flat = logits.view(-1, vocab_size)
    target_flat = target_ids.view(-1)
    mask_flat = mask_positions.reshape(-1)
    p_mask_flat = p_mask.reshape(-1)
    # Select masked positions
    masked_logits = logits_flat[mask_flat]
    masked_targets = target_flat[mask_flat]
    weights = 1.0 / (p_mask_flat[mask_flat] + epsilon)
    losses = nn.functional.cross_entropy(
        masked_logits, masked_targets, reduction="none"
    )
    return (losses * weights).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train masked-diffusion LLaDA POC")
    parser.add_argument(
        "--dataset_name", type=str, default="wikitext", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset config for HuggingFace datasets",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="Max sequence length"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load data and tokenizer
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.model_max_length = args.max_seq_len

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_len,
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model and optimizer
    config = DiffusionConfig(
        vocab_size=len(tokenizer),
        mask_token_id=tokenizer.mask_token_id,
        max_position_embeddings=args.max_seq_len,
    )
    model = DiffusionTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            noisy_inputs, mask_positions, p_mask = mask_inputs(
                input_ids, config.mask_token_id
            )
            logits = model(noisy_inputs, attention_mask)
            loss = compute_loss(logits, input_ids, mask_positions, p_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

    # Save the triumphant model
    save_path = os.path.join(args.model_dir, "model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

# LLaDA-poc

A minimal proof-of-concept implementation of a "masked-diffusion" language model (inspired by LLaDA). Instead of predicting text one token at a time, this model learns to reconstruct masked words and then generates text by iteratively filling and refining placeholders.

## How It Works

1. **Masked Prediction:** During training, we take real sentences and randomly hide (mask) words at different intensities. The model (a Transformer encoder) learns to guess the missing words.
2. **Diffusion Steps:** To generate text, we start with a prompt (or nothing) and a sequence of mask tokens. We perform multiple rounds (steps):
   - The model predicts tokens for all masked positions.
   - We sample from its predictions (using temperature, top-k, top-p).
   - Except on the final round, we re-mask a fraction of tokens to allow further refinement.
3. **Iterative Refinement:** Over several steps, the output goes from all blanks to a fully coherent sentence.

### Key Parameters

- **--steps:** Number of refinement rounds (more → finer results, slower).
- **--remask_ratio:** Fraction of tokens to mask again each round (e.g. 0.5 unmask half and re-mask half).
- **--temperature**, **--top_k**, **--top_p:** Control randomness in sampling to reduce repetition and improve diversity.

## Setup

1. Make sure you have **Python 3.8+** installed and (optionally) create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Project structure:

```text
.
├── src/
│   ├── model.py
│   ├── train.py
│   └── generate.py
├── requirements.txt
└── README.md
```

## Training

The `src/train.py` script will:

- Load and tokenize a text corpus (WikiText-2 by default).
- Randomly mask tokens to simulate diffusion noise.
- Train the Transformer to recover the original text.
- Save weights to `--model_dir`.

#### Example Command

```bash
python src/train.py \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --model_dir checkpoints \
  --batch_size 16 \
  --max_seq_len 128 \
  --epochs 1 \
  --learning_rate 5e-5
```

Model checkpoints will be saved to `checkpoints/model.pt`.

## Generation

Use `src/generate.py` to turn your trained model into a text generator. It will:

- Load your saved checkpoint.
- Optionally take a **prompt** to condition on.
- Perform multiple diffusion steps to fill and refine masked tokens.
- Print out the final generated text.

#### Example Command

```bash
python src/generate.py \
  --model_path checkpoints/model.pt \
  --tokenizer_name bert-base-uncased \
  --prompt "Once upon a time" \
  --max_seq_len 128 \
  --max_gen_len 50 \
  --steps 10 \
  --remask_ratio 0.5 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.9
```

## Notes

- Uses a `BertModel` as a bidirectional mask-predictor with a custom LM head.
- Training objective is a weighted cross-entropy only on masked tokens.
- Generation is an iterative mask-and-refine process, offering a speed-quality trade-off.

## License

MIT

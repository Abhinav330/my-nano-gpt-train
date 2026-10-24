"""
Evaluates HellaSwag in Python, loading the dataset from the Hugging Face Hub
(Rowan/hellaswag) instead of raw GitHub files.

Each HellaSwag example has a context (ctx) and 4 candidate endings; `label`
(0-3) is the index of the true ending. We score each ending by the
autoregressive loss over the ending tokens only, and pick the lowest loss.

Reference numbers for the validation split (10,042 examples), completion style:
gpt2 (124M):      acc: 0.2859  acc_norm: 0.2955
gpt2-xl (1558M):  acc: 0.3842  acc_norm: 0.4893

- acc:      picks the ending with the lowest SUMMED loss (favours short endings)
- acc_norm: picks the ending with the lowest AVERAGE per-token loss (length-normalised)
"""

import torch
import tiktoken
from torch.nn import functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Tokenizer. To use your own tokenizer, replace this with an object that has
# an `.encode(str) -> list[int]` method (keep the " " + ending handling below
# consistent with how your tokenizer treats leading spaces).
enc = tiktoken.get_encoding("gpt2")


def iterate_examples(split="validation"):
    """Yields HellaSwag examples from the Hugging Face Hub (cached locally)."""
    ds = load_dataset("Rowan/hellaswag", split=split)
    for example in ds:
        yield example


def render_example(example):
    """
    Given the example as a dictionary, render it as three things:
    - tokens (context + completion tokens, shape 4xN, as there are always 4 candidates)
    - mask (1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = int(example["label"])  # the Hub stores the label as a string
    endings = example["endings"]

    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # prepend " " because of the GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    # rows can differ in length, so pad to the longest
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label


def get_most_likely_row(tokens, mask, logits):
    """Returns (pred, pred_norm): the index of the lowest-loss ending, summed and averaged."""
    # evaluate the autoregressive loss at all positions
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # keep only the completion region (mask == 1); shift the mask too
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred = sum_loss.argmin().item()
    pred_norm = avg_loss.argmin().item()
    return pred, pred_norm, avg_loss


@torch.no_grad()
def evaluate(model_type, device, split="validation", limit=None):
    torch.set_float32_matmul_precision("high")  # use tf32
    # To evaluate your own model, load it here instead; it must return
    # an object with `.logits` of shape (B, T, vocab) when called on `tokens`.
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    model.eval()
    # model = torch.compile(model)  # optionally torch compile the model

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples(split):
        tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits
        pred, pred_norm, avg_loss = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(
            f"{num_total} acc: {num_correct}/{num_total}={num_correct/num_total:.4f} "
            f"acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}"
        )

        # debug: pretty print the first few examples and the loss for each ending
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print("Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

        if limit is not None and num_total >= limit:
            break

    print(f"\nFinal: {num_total} examples  "
          f"acc: {num_correct/num_total:.4f}  acc_norm: {num_correct_norm/num_total:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    parser.add_argument("-n", "--limit", type=int, default=None, help="only evaluate the first N examples")
    args = parser.parse_args()
    evaluate(args.model_type, args.device, limit=args.limit)
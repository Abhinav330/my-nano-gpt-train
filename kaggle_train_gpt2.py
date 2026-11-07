import os
import sys
import math
import time
import glob
import inspect
import threading
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Paths: work on Kaggle, Colab, local, notebook or script
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # notebook / interactive
    SCRIPT_DIR = os.getcwd()

# make `import hellaswag` work wherever hellaswag.py was placed
for p in (SCRIPT_DIR, os.getcwd(), "/kaggle/working"):
    if p not in sys.path:
        sys.path.insert(0, p)
from hellaswag import render_example, iterate_examples

# where logs/checkpoints are written
if os.environ.get("TRAIN_OUT_DIR"):
    OUT_DIR = os.environ["TRAIN_OUT_DIR"]
elif os.path.exists("/kaggle/working"):
    OUT_DIR = "/kaggle/working"
else:
    OUT_DIR = SCRIPT_DIR

def find_data_root(local_dir="edu_fineweb10B"):
    """Find the folder holding the edufineweb_*.npy shards."""
    candidates = []
    if os.environ.get("FINEWEB_OUT_DIR"):
        candidates.append(os.environ["FINEWEB_OUT_DIR"])
    candidates += [
        os.path.join("/kaggle/working", local_dir),
        os.path.join("/kaggle/temp", local_dir),
        os.path.join(SCRIPT_DIR, local_dir),
        os.path.join(os.getcwd(), local_dir),
    ]
    candidates += glob.glob(os.path.join("/kaggle/input", "*", local_dir))
    candidates += glob.glob(os.path.join("/kaggle/input", "*"))
    for c in candidates:
        if os.path.isdir(c) and any(f.startswith("edufineweb_") for f in os.listdir(c)):
            return c
    raise FileNotFoundError(
        "No edufineweb_*.npy shards found. Run fineweb.py first or set FINEWEB_OUT_DIR."
    )

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = find_data_root()
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split} in {data_root}"
        if master_process:
            print(f"found {len(shards)} shards for split {split} in {data_root}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    # --- for resuming after a session ends ---
    def get_state(self):
        return {"shard": self.current_shard,
                "base": self.current_position - self.B * self.T * self.process_rank}

    def set_state(self, state):
        self.current_shard = state["shard"] % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = state["base"] + self.B * self.T * self.process_rank

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous().float()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# simple launch:
# python kaggle_train_gpt2.py
# DDP launch for e.g. 4 GPUs:
# torchrun --standalone --nproc_per_node=4 kaggle_train_gpt2.py

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

# mixed precision: bf16 on Ampere+ GPUs, fp16 + GradScaler on older GPUs (T4/P100)
if device_type == "cuda" and torch.cuda.get_device_capability()[0] < 8:
    amp_dtype = torch.float16
else:
    amp_dtype = torch.bfloat16
use_scaler = (device_type == "cuda" and amp_dtype == torch.float16)
scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
if master_process:
    print(f"autocast dtype: {amp_dtype}, grad scaler: {use_scaler}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# ---- settings (override with env vars) ----
total_batch_size = int(os.environ.get("TOTAL_BATCH_SIZE", 524288)) # 2**19, ~0.5M tokens
T = 1024 # sequence length
B = int(os.environ.get("MICRO_BATCH", 8)) # micro batch size; lower it if you run out of GPU memory
eval_every = int(os.environ.get("EVAL_EVERY", 250))    # val loss, HellaSwag, samples
hella_limit = int(os.environ["HELLA_LIMIT"]) if os.environ.get("HELLA_LIMIT") else None
ckpt_every = int(os.environ.get("CKPT_EVERY", 500))    # rolling checkpoint (for resume)
save_model_every = int(os.environ.get("SAVE_MODEL_EVERY", 1000))  # permanent model_XXXXX.pt files
# torch.compile for the TRAINING forward/backward only. Eval/generation always run eager.
use_compile = device_type == "cuda" and os.environ.get("USE_COMPILE", "1") == "1"
# -------------------------------------------

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"micro batch size B={B}, T={T}")
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    print(f"torch.compile for training: {use_compile}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# EPOCHS overrides MAX_STEPS: one epoch = one pass over all train tokens
epochs = os.environ.get("EPOCHS")
if epochs:
    train_tokens = sum(np.load(s, mmap_mode="r").shape[0] for s in train_loader.shards)
    max_steps = max(1, int(float(epochs) * train_tokens / total_batch_size))
    if master_process:
        print(f"{train_tokens:,} train tokens, batch {total_batch_size:,} tokens "
              f"-> {max_steps} steps for {epochs} epoch(s)")
else:
    max_steps = int(os.environ.get("MAX_STEPS", 19073))
    if master_process:
        print(f"EPOCHS not set -> using MAX_STEPS={max_steps}")

torch.set_float32_matmul_precision('high')

# ---- model: one set of weights, two handles ----
#   raw_model   : plain eager model. Used for eval, HellaSwag, generation, checkpoints, optimizer.
#   train_model : compiled (+ DDP) wrapper around the SAME weights. Used only for training steps.
raw_model = GPT(GPTConfig(vocab_size=50304))
raw_model.to(device)
train_model = torch.compile(raw_model) if use_compile else raw_model
if ddp:
    train_model = DDP(train_model, device_ids=[ddp_local_rank])

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = min(715, max_steps // 10)
def get_lr(it):
    # 1) linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) after the schedule ends, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# log directory for checkpoints and logs
log_dir = os.path.join(OUT_DIR, "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
latest_path = os.path.join(log_dir, "latest.pt")

def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

# ---------------- Hugging Face checkpoint sync ----------------
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")   # e.g. "your-username/gpt2-124m-fineweb"
hf_api = None
if HF_TOKEN and HF_REPO_ID:
    from huggingface_hub import HfApi
    hf_api = HfApi(token=HF_TOKEN)
    try:
        hf_api.create_repo(HF_REPO_ID, repo_type="model", private=True, exist_ok=True)
    except Exception as e:
        print(f"HF repo setup failed ({e}); HF sync disabled")
        hf_api = None
elif master_process:
    print("HF_TOKEN / HF_REPO_ID not set -> checkpoints stay local only")

_upload_thread = None
_n_uploads = 0

def wait_for_upload():
    global _upload_thread
    if _upload_thread is not None:
        _upload_thread.join()
        _upload_thread = None

def push_logs():
    if hf_api is None:
        return
    try:
        if os.path.exists(log_file):
            hf_api.upload_file(path_or_fileobj=log_file, path_in_repo="log.txt", repo_id=HF_REPO_ID)
        runs_dir = os.path.join(OUT_DIR, "runs")
        if os.path.isdir(runs_dir):
            hf_api.upload_folder(folder_path=runs_dir, path_in_repo="runs", repo_id=HF_REPO_ID)
    except Exception as e:
        print(f"HF log upload failed: {e}")

def _upload_and_cleanup(local_path, repo_path, step):
    global _n_uploads
    try:
        hf_api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_path, repo_id=HF_REPO_ID,
                           commit_message=f"{repo_path} @ step {step}")
        os.remove(local_path)
        print(f"[HF] uploaded {repo_path} (step {step}) and deleted local copy", flush=True)
        _n_uploads += 1
        if _n_uploads % 5 == 0:
            hf_api.super_squash_history(repo_id=HF_REPO_ID)
        push_logs()
    except Exception as e:
        print(f"[HF] upload failed ({e}); keeping local file {local_path}", flush=True)

def save_checkpoint(obj, local_path, repo_path, step):
    global _upload_thread
    wait_for_upload()
    atomic_save(obj, local_path)
    if hf_api is None:
        return
    _upload_thread = threading.Thread(target=_upload_and_cleanup, args=(local_path, repo_path, step))
    _upload_thread.start()
# ----------------------------------------------------------------

# resume priority: RESUME_FROM env var -> local latest.pt -> latest.pt on Hugging Face -> fresh run
start_step = 0
resume_path = os.environ.get("RESUME_FROM")
if not resume_path:
    if os.path.exists(latest_path):
        resume_path = latest_path
    elif hf_api is not None and os.environ.get("HF_RESUME", "1") == "1":
        try:
            from huggingface_hub import hf_hub_download
            resume_path = hf_hub_download(HF_REPO_ID, "latest.pt", token=HF_TOKEN)
            if master_process:
                print(f"found latest.pt on Hugging Face ({HF_REPO_ID}), resuming from it")
        except Exception:
            resume_path = None
if resume_path and os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if use_scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    train_loader.set_state(ckpt["train_loader"])
    start_step = ckpt["step"]
    if master_process:
        print(f"resumed from {resume_path} at step {start_step}")
    del ckpt
elif master_process:
    with open(log_file, "w") as f:
        pass

# TensorBoard (master process only)
writer = None
if master_process:
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(OUT_DIR, "runs"),
                               purge_step=start_step if start_step > 0 else None)
        print(f"TensorBoard logs: {os.path.join(OUT_DIR, 'runs')}")
    except Exception as e:
        print(f"TensorBoard disabled ({e}). Run: pip install tensorboard")

def tb(tag, value, step):
    if writer is not None:
        writer.add_scalar(tag, value, step)

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss (eager model)
    if step % eval_every == 0 or last_step:
        raw_model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = raw_model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}", flush=True)
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            tb("val/loss", val_loss_accum.item(), step)
            if step > 0 and (step % save_model_every == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                save_checkpoint(checkpoint, checkpoint_path, f"model_{step:05d}.pt", step)

    # once in a while evaluate hellaswag (eager model: variable shapes would recompile a compiled one)
    if step % eval_every == 0 or last_step:
        raw_model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("validation")):
            if hella_limit is not None and i >= hella_limit:
                break
            if i % ddp_world_size != ddp_rank:
                continue
            tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = raw_model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}", flush=True)
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
            tb("val/hellaswag_acc_norm", acc_norm, step)

    # once in a while generate from the model (eager model; skip step 0)
    if (step > 0 and step % eval_every == 0) or last_step:
        raw_model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = raw_model(xgen)
                logits = logits[:, -1, :].float()
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}", flush=True)
            if master_process and writer is not None and i == 0:
                writer.add_text("samples/generation", decoded, step)

    # rolling checkpoint (model + optimizer + data position); uploaded to HF then deleted locally
    if master_process and step > start_step and step % ckpt_every == 0:
        save_checkpoint({
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if use_scaler else None,
            'config': raw_model.config,
            'step': step,
            'train_loader': train_loader.get_state(),
        }, latest_path, "latest.pt", step)

    # do one step of the optimization (compiled model)
    raw_model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            train_model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            logits, loss = train_model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        scaler.scale(loss).backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    scaler.step(optimizer)
    scaler.update()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}", flush=True)
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
        tb("train/loss", loss_accum.item(), step)
        tb("train/lr", lr, step)
        tb("train/grad_norm", float(norm), step)
        tb("train/tok_per_sec", tokens_per_sec, step)
        if writer is not None and step % 50 == 0:
            writer.flush()

if master_process:
    if writer is not None:
        writer.close()
    wait_for_upload()
    push_logs()
if ddp:
    destroy_process_group()
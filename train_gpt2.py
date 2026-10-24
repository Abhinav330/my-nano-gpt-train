import os
import json
import math
import time
import atexit
import queue
import shutil
import inspect
import threading
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
from safetensors.torch import save_model, load_model, save_file, load_file
from hellaswag import render_example, iterate_examples

# load HF_TOKEN / HF_REPO_ID from the .env file (never commit or upload .env)
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")  # e.g. "your-username/gpt2-124m-run"
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 1000))            # full resumable checkpoint (model+optimizer+data position)
MILESTONE_EVERY = int(os.environ.get("MILESTONE_EVERY", 5000))  # extra model-only snapshot that is kept in the repo
HIST_EVERY = int(os.environ.get("HIST_EVERY", 1000))            # weight/grad histograms in TensorBoard (0 = off)
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
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
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
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

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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
        # Create AdamW optimizer and use the fused version if it is available
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
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    # --- added: save / restore the data position so a resumed run continues where it stopped
    def state_dict(self):
        return {
            'current_shard': self.current_shard,
            # store the position without this rank's offset, so it works for any rank
            'base_position': self.current_position - self.B * self.T * self.process_rank,
        }

    def load_state_dict(self, sd):
        self.current_shard = sd['current_shard']
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = sd['base_position'] + self.B * self.T * self.process_rank

# -----------------------------------------------------------------------------
# added: background uploader that pushes files to a (private) Hugging Face repo
# Uploads run in a separate thread so training never waits for the network.

from huggingface_hub import HfApi, CommitOperationAdd

class HFUploader:
    def __init__(self, repo_id, token, enabled=True):
        self.enabled = bool(enabled and repo_id and token)
        self.repo_id = repo_id
        self.q = queue.Queue()
        if not self.enabled:
            if enabled:
                print("HF upload disabled: set HF_TOKEN and HF_REPO_ID to enable it")
            return
        self.api = HfApi(token=token)
        self.api.create_repo(repo_id, private=True, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()
        atexit.register(self.wait)  # flush pending uploads on exit
        print(f"HF upload enabled -> {repo_id}")

    def push(self, files, message):
        """files: list of (local_path, path_in_repo). Uploaded together in one commit."""
        if self.enabled:
            self.q.put((files, message))

    def wait(self):
        if self.enabled:
            self.q.join()

    def _worker(self):
        while True:
            files, message = self.q.get()
            try:
                for attempt in range(3):
                    try:
                        ops = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=lp)
                               for lp, rp in files if os.path.exists(lp)]
                        self.api.create_commit(repo_id=self.repo_id, operations=ops, commit_message=message)
                        print(f"[hf] pushed: {message}")
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        time.sleep(10 * (attempt + 1))
            except Exception as e:
                print(f"[hf] upload failed (training continues): {e}")
            finally:
                self.q.task_done()

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# resume a run (from log/ if present, otherwise downloaded from the HF repo):
# RESUME=1 python train_gpt2.py
# view the curves:  tensorboard --logdir log/tb

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
tb_dir = os.path.join(log_dir, "tb")

# only the master process talks to Hugging Face
uploader = HFUploader(HF_REPO_ID, HF_TOKEN, enabled=master_process)

# ---- optional resume (RESUME=1): continue from the last full checkpoint ----
start_step = 0
if os.environ.get("RESUME", "0") == "1":
    names = ["latest_model.safetensors", "latest_optimizer.safetensors", "latest_state.json"]
    paths = {n: os.path.join(log_dir, n) for n in names}
    if not all(os.path.exists(p) for p in paths.values()) and HF_REPO_ID and HF_TOKEN:
        try:
            from huggingface_hub import hf_hub_download
            paths = {n: hf_hub_download(HF_REPO_ID, f"checkpoints/{n}", token=HF_TOKEN) for n in names}
        except Exception as e:
            print(f"could not download checkpoint from HF: {e}")
            paths = None
    if paths and all(os.path.exists(p) for p in paths.values()):
        load_model(raw_model, paths["latest_model.safetensors"])
        state = json.load(open(paths["latest_state.json"]))
        flat = load_file(paths["latest_optimizer.safetensors"], device=device)
        opt_state = {}
        for key, t in flat.items():
            i, k = key.split(".", 1)
            opt_state.setdefault(int(i), {})[k] = t
        optimizer.load_state_dict({'state': opt_state, 'param_groups': state['param_groups']})
        train_loader.load_state_dict(state['train_loader'])
        start_step = state['step']
        if master_process:
            print(f"resumed at step {start_step}")
        # if the old TensorBoard events are not on disk (fresh server), pull them from the HF repo
        if master_process and not os.path.isdir(tb_dir) and HF_REPO_ID and HF_TOKEN:
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(HF_REPO_ID, allow_patterns=["tb/*", "log/log.txt"],
                                  local_dir=log_dir, token=HF_TOKEN)
                # files land in log/tb/... and log/log/log.txt; move the text log into place
                nested = os.path.join(log_dir, "log", "log.txt")
                if os.path.exists(nested) and not os.path.exists(log_file):
                    shutil.move(nested, log_file)
            except Exception as e:
                print(f"could not restore previous TensorBoard logs: {e}")
    elif master_process:
        print("RESUME=1 but no checkpoint found, starting from scratch")

if start_step == 0 and master_process:
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    shutil.rmtree(tb_dir, ignore_errors=True) # fresh run -> fresh TensorBoard logs

# TensorBoard writer (master only). On resume, purge_step drops events from steps that will be re-run.
writer = None
if master_process:
    writer = SummaryWriter(tb_dir, purge_step=start_step if start_step > 0 else None)
    if start_step == 0:
        hparams = dict(total_batch_size=total_batch_size, micro_batch=B, seq_len=T, grad_accum_steps=grad_accum_steps,
                       max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, max_steps=max_steps,
                       weight_decay=0.1, world_size=ddp_world_size, model=asdict(raw_model.config))
        writer.add_text("hparams", "```\n" + json.dumps(hparams, indent=2) + "\n```", 0)

def tb_files():
    """All TensorBoard event files as (local_path, path_in_repo). The HF Hub shows them in its metrics tab."""
    out = []
    for root, _, fs in os.walk(tb_dir):
        for f in fs:
            lp = os.path.join(root, f)
            out.append((lp, "tb/" + os.path.relpath(lp, tb_dir).replace("\\", "/")))
    return out

# push a snapshot of the code once at the start (never .env)
if master_process:
    here = os.path.dirname(os.path.abspath(__file__))
    code_files = [(os.path.join(here, n), n) for n in ("train_gpt2.py", "hellaswag.py", "requirements.txt")]
    uploader.push(code_files, "code snapshot")

def _atomic(write_fn, path):
    """Write to path.tmp, then rename, so a crash never leaves a half-written file."""
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)

def save_checkpoint(next_step, milestone, val_loss):
    """Save model + optimizer as safetensors (atomic), then push to the HF repo along with logs."""
    uploader.wait()  # previous upload must finish before we overwrite files
    model_p = os.path.join(log_dir, "latest_model.safetensors")
    opt_p   = os.path.join(log_dir, "latest_optimizer.safetensors")
    state_p = os.path.join(log_dir, "latest_state.json")

    # model (save_model handles the tied wte/lm_head weights)
    _atomic(lambda p: save_model(raw_model, p), model_p)

    # optimizer: tensors -> safetensors, everything else -> json
    osd = optimizer.state_dict()
    flat = {f"{i}.{k}": (v if torch.is_tensor(v) else torch.tensor(v)).detach().cpu().contiguous()
            for i, s in osd['state'].items() for k, v in s.items()}
    _atomic(lambda p: save_file(flat, p), opt_p)

    state = {'step': next_step, 'val_loss': val_loss,
             'config': asdict(raw_model.config),
             'param_groups': osd['param_groups'],
             'train_loader': train_loader.state_dict()}
    def _write_json(p):
        with open(p, "w") as f:
            f.write(json.dumps(state))
    _atomic(_write_json, state_p)

    writer.flush()  # make sure the event files on disk are up to date before uploading
    files = [(model_p, "checkpoints/latest_model.safetensors"),
             (opt_p,   "checkpoints/latest_optimizer.safetensors"),
             (state_p, "checkpoints/latest_state.json"),
             (log_file, "log/log.txt")] + tb_files()
    if milestone:  # model-only snapshot, kept in the repo
        mp = os.path.join(log_dir, f"model_{next_step:05d}.safetensors")
        _atomic(lambda p: save_model(raw_model, p), mp)
        files.append((mp, f"checkpoints/model_{next_step:05d}.safetensors"))
    uploader.push(files, f"step {next_step}")

last_val_loss = None

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        last_val_loss = val_loss_accum.item()
        if master_process:
            print(f"validation loss: {last_val_loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {last_val_loss:.4f}\n")
            writer.add_scalar("val/loss", last_val_loss, step)
            writer.add_scalar("val/loss_vs_tokens", last_val_loss, step * total_batch_size)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("validation")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
            writer.add_scalar("eval/hellaswag_acc_norm", acc_norm, step)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        samples = []
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
            samples.append(f"{i}: {decoded}")
        if master_process:
            writer.add_text("samples", "\n\n".join(samples), step)

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # histograms of weights and gradients (before the update, so the grads are still there)
    if master_process and HIST_EVERY > 0 and step % HIST_EVERY == 0:
        for name, p in raw_model.named_parameters():
            writer.add_histogram(f"weights/{name}", p.detach().float().cpu(), step)
            if p.grad is not None:
                writer.add_histogram(f"grads/{name}", p.grad.detach().float().cpu(), step)

    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
            f.write(f"{step} lr {lr:.6e}\n")
            f.write(f"{step} norm {norm:.4f}\n")
            f.write(f"{step} tps {tokens_per_sec:.0f}\n")
        # TensorBoard scalars
        writer.add_scalar("train/loss", loss_accum.item(), step)
        writer.add_scalar("train/lr", lr, step)
        writer.add_scalar("train/grad_norm", float(norm), step)
        writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)
        writer.add_scalar("train/step_time_ms", dt * 1000, step)
        writer.add_scalar("train/tokens_seen", (step + 1) * total_batch_size, step)
        if device_type == "cuda":
            writer.add_scalar("system/gpu_mem_gb", torch.cuda.max_memory_allocated() / 1e9, step)

    # save a resumable checkpoint (and push it to Hugging Face) every CKPT_EVERY steps and at the end
    if master_process and ((step + 1) % CKPT_EVERY == 0 or last_step):
        milestone = ((step + 1) % MILESTONE_EVERY == 0) or last_step
        save_checkpoint(step + 1, milestone, last_val_loss)

if master_process:
    writer.flush()
    writer.close()
    # final push of the logs in case the last step was not a checkpoint step
    uploader.push([(log_file, "log/log.txt")] + tb_files(), "final logs")
uploader.wait()  # make sure the final checkpoint and logs have been uploaded
if ddp:
    destroy_process_group()
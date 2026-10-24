#!/usr/bin/env bash
cd "$(dirname "$0")"

# install uv if it's missing
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# venv that reuses the template's CUDA torch
uv venv --system-site-packages

# install the rest (torch is already satisfied)
uv pip install --python .venv/bin/python -r requirements.txt

# sanity check
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
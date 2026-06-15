import numpy as np
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def stable_jsd_squared(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        kl_pm = np.where(p > 0, p * np.log2(np.where(m > 0, p / m, 1.0)), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(np.where(m > 0, q / m, 1.0)), 0.0)
    return max(0.0, float(0.5 * kl_pm.sum() + 0.5 * kl_qm.sum()))


def load_config(config_name: str) -> dict:
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    with open(config_dir / config_name) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_key: str, device_map: str = "auto"):
    cfg = load_config("model_config.yaml")
    model_cfg = cfg["models"][model_key]
    hf_id = model_cfg["hf_id"]
    dtype = getattr(torch, cfg["architecture"]["dtype"])

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map=device_map,
        torch_dtype=dtype,
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer


def check_tied_embeddings(model) -> bool:
    return model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def get_layer_groups() -> dict:
    cfg = load_config("model_config.yaml")
    return cfg["layer_groups"]


def get_num_layers() -> int:
    cfg = load_config("model_config.yaml")
    return cfg["architecture"]["num_layers"]


def extract_hidden_state(model, tokenizer, text: str, layer: int, position: str = "final"):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    hs = outputs.hidden_states[layer]  # [1, seq_len, hidden_size]

    if position == "final":
        return hs[0, -1, :]
    elif position == "mean":
        return hs[0].mean(dim=0)
    elif isinstance(position, tuple):
        start, end = position
        return hs[0, start:end + 1, :].mean(dim=0)
    else:
        raise ValueError(f"Unknown position: {position}")


def extract_all_layers(model, tokenizer, text: str, position: str = "final"):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    results = []
    for layer_hs in outputs.hidden_states:
        if position == "final":
            results.append(layer_hs[0, -1, :])
        elif position == "mean":
            results.append(layer_hs[0].mean(dim=0))
        else:
            raise ValueError(f"Unknown position: {position}")
    return results

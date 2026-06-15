import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils import load_model_and_tokenizer, get_num_layers


RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

SEMANTIC_BINS = {
    "hierarchical": ["hier", "archical", "hierarchy", "階層", "的"],
    "egalitarian": ["equal", "egalitarian", "平等", "な"],
    "male_dominated": ["male", "domin", "男性", "中心"],
    "positive_adjective": ["good", "great", "nice", "excellent", "良い", "素晴らしい", "優れた"],
    "negative_adjective": ["bad", "poor", "terrible", "awful", "悪い", "ひどい", "酷い"],
    "old_age": ["old", "elderly", "senior", "aged", "高齢", "年配", "老"],
    "young_age": ["young", "youth", "junior", "若い", "若者", "青年"],
    "ability": ["able", "capable", "competent", "skilled", "有能", "能力"],
    "disability": ["disabled", "handicapped", "impaired", "障害", "不自由"],
    "male": ["he", "him", "his", "man", "boy", "彼", "男", "男性"],
    "female": ["she", "her", "woman", "girl", "彼女", "女", "女性"],
    "positive_trait": ["kind", "smart", "honest", "brave", "優しい", "賢い", "正直"],
    "negative_trait": ["lazy", "stupid", "dishonest", "cruel", "怠惰", "愚か", "不正直"],
}


def logit_lens(model, hidden_state: torch.Tensor) -> torch.Tensor:
    logits = model.lm_head(hidden_state.unsqueeze(0).to(model.lm_head.weight.dtype))
    log_probs = torch.log_softmax(logits.float(), dim=-1).squeeze(0)
    return log_probs


def binned_distribution(log_probs: torch.Tensor, semantic_bins: dict,
                         tokenizer) -> dict:
    bin_probs = {}
    for bin_name, terms in semantic_bins.items():
        total_prob = 0.0
        for term in terms:
            token_ids = tokenizer.encode(term, add_special_tokens=False)
            for tid in token_ids:
                if tid < len(log_probs):
                    total_prob += torch.exp(log_probs[tid]).item()
        bin_probs[bin_name] = total_prob

    total = sum(bin_probs.values())
    if total > 0:
        bin_probs = {k: v / total for k, v in bin_probs.items()}
    return bin_probs


def top_k_tokens(log_probs: torch.Tensor, tokenizer, k: int = 100) -> list:
    top_indices = torch.topk(log_probs, k).indices.tolist()
    return [(tokenizer.decode([idx]), log_probs[idx].item()) for idx in top_indices]


def extract_logit_lens_single(model, tokenizer, text: str,
                                layers: list[int] = None) -> dict:
    num_layers = get_num_layers()
    if layers is None:
        layers = list(range(num_layers + 1))

    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    result = {}
    for layer in layers:
        hs = outputs.hidden_states[layer][0, -1, :]
        lp = logit_lens(model, hs)
        bins = binned_distribution(lp, SEMANTIC_BINS, tokenizer)
        top_k = top_k_tokens(lp, tokenizer, k=100)
        result[layer] = {"binned": bins, "top_k": top_k}

    return result


def extract_logit_lens_pairs(model_key: str, paired_templates: list,
                              layers: list[int] = None):
    model, tokenizer = load_model_and_tokenizer(model_key)
    num_layers = get_num_layers()
    if layers is None:
        layers = list(range(num_layers + 1))

    all_results = []
    for pair in tqdm(paired_templates, desc="Extracting logit lens"):
        ll_en = extract_logit_lens_single(model, tokenizer, pair["context_en"], layers)
        ll_ja = extract_logit_lens_single(model, tokenizer, pair["context_ja"], layers)
        all_results.append({
            "category": pair["category"],
            "fidelity": pair.get("fidelity", None),
            "logit_lens_en": {
                str(l): d["binned"] for l, d in ll_en.items()
            },
            "logit_lens_ja": {
                str(l): d["binned"] for l, d in ll_ja.items()
            },
            "top_k_en": {
                str(l): d["top_k"] for l, d in ll_en.items()
            },
            "top_k_ja": {
                str(l): d["top_k"] for l, d in ll_ja.items()
            },
        })

    out_dir = RESULTS_DIR / "exp2" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(out_dir / "logit_lens.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    del model
    torch.cuda.empty_cache()

    return all_results

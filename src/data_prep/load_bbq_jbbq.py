import json
import csv
from pathlib import Path
from collections import defaultdict


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SHARED_CATEGORIES = ["Age", "Disability_status", "Gender_identity", "Physical_appearance", "Sexual_orientation"]


def load_bbq(bbq_dir: Path = None) -> dict:
    bbq_dir = bbq_dir or DATA_DIR / "bbq" / "data"
    templates = defaultdict(list)

    for jsonl_file in sorted(bbq_dir.glob("*.jsonl")):
        category = jsonl_file.stem
        if category not in SHARED_CATEGORIES:
            continue
        with open(jsonl_file) as f:
            for line in f:
                item = json.loads(line.strip())
                item["category"] = category
                templates[category].append(item)

    return dict(templates)


def load_jbbq(jbbq_dir: Path = None) -> dict:
    jbbq_dir = jbbq_dir or DATA_DIR / "jbbq"
    templates = defaultdict(list)

    for jsonl_file in sorted(jbbq_dir.glob("*.jsonl")):
        category = jsonl_file.stem
        if category not in SHARED_CATEGORIES:
            continue
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                item["category"] = category
                templates[category].append(item)

    return dict(templates)


def group_by_template(items: list, template_key: str = "template_id") -> dict:
    grouped = defaultdict(list)
    for item in items:
        tid = item.get(template_key, item.get("example_id", "unknown"))
        grouped[tid].append(item)
    return dict(grouped)


def filter_ambiguous(items: list) -> list:
    return [item for item in items if item.get("context_condition") == "ambig"
            or item.get("context_type", "").startswith("ambig")]


def extract_contexts(items: list) -> list:
    contexts = []
    for item in items:
        ctx = item.get("context", item.get("context_ja", ""))
        if ctx:
            contexts.append(ctx)
    return contexts


def sample_instances_per_template(grouped: dict, n: int = 3, seed: int = 42) -> dict:
    import random
    rng = random.Random(seed)
    sampled = {}
    for tid, instances in grouped.items():
        if len(instances) <= n:
            sampled[tid] = instances
        else:
            sampled[tid] = rng.sample(instances, n)
    return sampled


def sample_paired_templates(paired: list, max_per_context: int = 3, seed: int = 42) -> list:
    import random
    rng = random.Random(seed)
    by_context = defaultdict(list)
    for p in paired:
        by_context[p["context_en"]].append(p)
    sampled = []
    for ctx, items in by_context.items():
        items_sorted = sorted(items, key=lambda x: x.get("fidelity", 0), reverse=True)
        sampled.extend(items_sorted[:max_per_context])
    rng.shuffle(sampled)
    return sampled

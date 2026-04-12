import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


@dataclass
class ModelSpec:
    name: str
    tag: str
    mode: str # st для sentence transformer, bert_meanpool для обычного BERT


def load_segments(json_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segs = data.get("segments", [])
    texts = []
    meta = []

    for s in segs:
        txt = (s.get("text") or "").strip()
        texts.append(txt)

        meta.append(
            {
                "id": str(s.get("id", "")).lower().strip(),
                "role": (s.get("role") or "").strip(),
                "path": (s.get("path") or "").strip(),
                "parent": str(s.get("parent") or "").strip(),
            }
        )

    return texts, meta


def save_meta(meta: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def encode_bert_meanpool(model_name: str, texts: List[str], device: str, batch_size: int = 8) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()

    parts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = mdl(**enc)
        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        parts.append(pooled.detach().cpu().numpy())

    return np.vstack(parts).astype(np.float32)


def encode_sentence_transformer(model_name: str, texts: List[str], device: str, batch_size: int = 16) -> np.ndarray:
    st = SentenceTransformer(model_name, device=device)
    emb = st.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def get_default_models() -> List[ModelSpec]:
    return [
        ModelSpec("ai-forever/FRIDA", "FRIDA", "st"),
        ModelSpec("ai-forever/sbert_large_nlu_ru", "sbert_large_nlu_ru", "st"),
        ModelSpec("ai-forever/ruBERT-base", "rubert_base", "bert_meanpool"),
        ModelSpec("cointegrated/rubert-tiny2", "rubert_tiny2", "bert_meanpool"),
        ModelSpec("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "multi_mpnet_v2", "st"),
        ModelSpec("sentence-transformers/all-mpnet-base-v2", "all_mpnet_v2", "st"),
    ]


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_embeddings_for_models(
    texts: List[str],
    models: List[ModelSpec],
    out_dir: str,
    device: str | None = None,
    batch_size_cpu: int = 8,
    batch_size_gpu: int = 16,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = pick_device()

    saved = {}

    for m in models:
        if m.mode == "st":
            bs = batch_size_gpu if device == "cuda" else batch_size_cpu
            vecs = encode_sentence_transformer(m.name, texts, device=device, batch_size=bs)
        else:
            bs = batch_size_cpu if device != "cuda" else max(8, batch_size_gpu // 2)
            vecs = encode_bert_meanpool(m.name, texts, device=device, batch_size=bs)

        path = os.path.join(out_dir, f"emb_{m.tag}.npy")
        np.save(path, vecs)
        saved[m.tag] = path

    return saved


def prepare_embeddings_pack(
    input_json: str,
    out_dir: str = "out",
    models: List[ModelSpec] | None = None,
    device: str | None = None,
) -> Dict[str, Any]:
    texts, meta = load_segments(input_json)

    if models is None:
        models = get_default_models()

    os.makedirs(out_dir, exist_ok=True)
    save_meta(meta, os.path.join(out_dir, "meta.json"))

    saved = compute_embeddings_for_models(
        texts=texts,
        models=models,
        out_dir=out_dir,
        device=device,
    )

    return {
        "out_dir": out_dir,
        "device": device or pick_device(),
        "count": len(texts),
        "meta_path": os.path.join(out_dir, "meta.json"),
        "embeddings": saved,
    }


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def load_meta(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

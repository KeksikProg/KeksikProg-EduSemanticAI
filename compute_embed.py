import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


@dataclass
class ModelSpec:
    name: str
    tag: str
    mode: str


def load_segments(json_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    for segment in segments:
        texts.append((segment.get("text") or "").strip())
        meta.append(
            {
                "id": str(segment.get("id", "")).lower().strip(),
                "role": (segment.get("role") or "").strip(),
                "path": (segment.get("path") or "").strip(),
                "parent": str(segment.get("parent") or "").strip(),
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    parts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        output = model(**encoded)
        pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        parts.append(pooled.detach().cpu().numpy())

    return np.vstack(parts).astype(np.float32)


def encode_sentence_transformer(model_name: str, texts: List[str], device: str, batch_size: int = 16) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def get_default_models() -> List[ModelSpec]:
    return [
        ModelSpec("ai-forever/FRIDA", "FRIDA", "st"),
        # ModelSpec("ai-forever/sbert_large_nlu_ru", "sbert_large_nlu_ru", "st"),
        # ModelSpec("ai-forever/ruBERT-base", "rubert_base", "bert_meanpool"),
        # ModelSpec("cointegrated/rubert-tiny2", "rubert_tiny2", "bert_meanpool"),
        # ModelSpec("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "multi_mpnet_v2", "st"),
        # ModelSpec("sentence-transformers/all-mpnet-base-v2", "all_mpnet_v2", "st"),
    ]


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _compute_embeddings_for_model(
    model: ModelSpec,
    texts: List[str],
    device: str,
    batch_size_cpu: int,
    batch_size_gpu: int,
) -> np.ndarray:
    if model.mode == "st":
        batch_size = batch_size_gpu if device == "cuda" else batch_size_cpu
        return encode_sentence_transformer(model.name, texts, device=device, batch_size=batch_size)

    batch_size = batch_size_cpu if device != "cuda" else max(8, batch_size_gpu // 2)
    return encode_bert_meanpool(model.name, texts, device=device, batch_size=batch_size)


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

    saved: Dict[str, str] = {}
    for model in models:
        vectors = _compute_embeddings_for_model(
            model=model,
            texts=texts,
            device=device,
            batch_size_cpu=batch_size_cpu,
            batch_size_gpu=batch_size_gpu,
        )

        path = os.path.join(out_dir, f"emb_{model.tag}.npy")
        np.save(path, vectors)
        saved[model.tag] = path

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
    meta_path = os.path.join(out_dir, "meta.json")
    save_meta(meta, meta_path)

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
        "meta_path": meta_path,
        "embeddings": saved,
    }


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def load_meta(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

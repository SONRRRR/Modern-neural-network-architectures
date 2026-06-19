import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_device = None
_tokenizer = None
_model = None

def get_model():
    """Инициализация модели"""
    global _device, _tokenizer, _model
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _model = AutoModelForCausalLM.from_pretrained("gpt2").to(_device)
        _model.eval()
    return _device, _tokenizer, _model

def score(text: str) -> float:
    "Вычисление энтропии одного текста"
    device, tokenizer, model = get_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=input_ids).loss
    return float(loss.item())

def rec_entropy(records):
    """Добавление поля со значением энтропии к каждой записи"""
    for rcrd in records:
        rcrd["entropy"] = score(rcrd["text"])
        yield rcrd

def remove_dupl(records):
    """Удаление записей с одинаковыми текстами"""
    seen = set()
    for rcrd in records:
        key = " ".join(rcrd["text"].lower().split())
        if key not in seen:
            seen.add(key)
            yield rcrd

def filter_entropy(records, low=0.05, high=0.95):
    """Удаление записей с экстремальной энтропией"""
    records = list(records)
    vals = [r["entropy"] for r in records if not np.isnan(r["entropy"])]
    if not vals:
        return []
    qlow, qhigh = np.quantile(vals, [low, high])
    return [r for r in records if qlow <= r.get("entropy", -np.inf) <= qhigh]

def info_density(records):
    """Вычисление информационной плотности всего датасета"""
    tokens = [len(r["text"].split()) for r in records]
    entropies = [r["entropy"] for r in records]
    
      # Взвешенное среднее = сумма(энтропия * вес) / сумма(весов)
    return np.average(entropies, weights=tokens)
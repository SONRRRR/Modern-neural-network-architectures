from collections.abc import Iterator

from datasets import load_dataset


def get_text(split: str = "train", name: str = "wikitext-2-raw-v1", limit: int | None = None) -> Iterator[dict]:
    dataset = load_dataset("wikitext", name, split=split)
    emitted = 0
    for index, row in enumerate(dataset):
        text = row.get("text", "")
        if text and text.strip():
            yield {"id": f"wikitext-{split}-{index}", "url": None, "text": text}
            emitted += 1
            if limit is not None and emitted >= limit:
                return
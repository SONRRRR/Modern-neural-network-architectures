import os, click, json
import numpy as np
from pathlib import Path
from datasets import load_dataset
 
from src.download_warc import download_warc
from src.download_wikitext import get_text
from src.warc2text import warc2jsonl
from src.cleaning_text import text_process
from src.entropy_filter import rec_entropy, remove_dupl, filter_entropy, info_density
from src.tokenization import tokenizer
from src.packed_batching import packing

CC_SNAPSHOT = "CC-MAIN-2014-15" # CC-MAIN-2024-42
DATA_DIR = Path(__file__).parent.parent / "data"

@click.group()
def cli():
    """Lab 1 data preparation CLI"""
    pass

@cli.command()
@click.option("--snapshot", default=CC_SNAPSHOT, help="Common Crawl snapshot name")
@click.option("--warc-index", default=0, type=int, help="WARC file index in manifest")
@click.option("--output", default=None, help="Output path for downloaded file")
@click.option("--smallest", is_flag=True, default=False, help="Download smallest WARC")

def download_cc(
    snapshot = CC_SNAPSHOT, 
    warc_index = 0, 
    output: str = None, 
    smallest: bool = False
    ) -> None:

    paths_url = f"https://data.commoncrawl.org/crawl-data/{snapshot}/warc.paths.gz"
    
    if output is None:
        output = str(DATA_DIR / f"{snapshot}_index{warc_index}.warc.gz")

    path = download_warc(
    paths_url=paths_url,
    output_path=output,
    smallest=smallest,
    warc_index=warc_index)

    print(f"Downloaded WARC: {path}")       
 
@cli.command()
@click.option("--split", default="train", help="Split: train/validation/test")
@click.option("--limit", type=int, default=None, help="Limit number of records")
@click.option("--output", default=None, help="Output JSONL file path")
def download_wt(split, limit, output):
    """Загрузка WikiText-2 в формате JSONL"""
    
    if output is None:
        output = str(DATA_DIR / f"wikitext_{split}.jsonl")
    
    print(f"Загрузка WikiText-2 ({split} split)...")
    print(f"Лимит: {limit if limit else 'все записи'}")
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for record in get_text(split=split, limit=limit):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    
    print(f"Сохранено {count} записей -> {output}")

@cli.command()
@click.option("--input", required=True, help="Input WARC file")
@click.option("--output", default=None, help="Output JSONL file (auto-generated if not specified)")
@click.option("--limit", type=int, default=None, help="Limit records for testing")
def warc_to_jsonl(input, output, limit):
    count = warc2jsonl(input, output, limit)
    if count == 0:
        print("Не удалось извлечь ни одной записи")
    else:
        print(f"Конвертировано {count} записей")

@cli.command()
@click.option("--input", required=True)
@click.option("--output", required=True)
@click.option("--languages", default="en,ru")
@click.option("--min-words", default=50, type=int)
@click.option("--max-words", default=700, type=int)
def clean(input, output, languages, min_words, max_words):    
    lang_list = [lang.strip() for lang in languages.split(",")]
    
    print(f"Очистка: {input} -> {output}")
    print(f"  Языки: {lang_list}, длинный текст разбит по {max_words} слов")
    records = []
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    cleaned = list(text_process(records, languages=lang_list, min_words=min_words, max_words=max_words))
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w", encoding="utf-8") as f:
        for record in cleaned:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Очищено записей: {len(cleaned)}")


@cli.command()
@click.option("--input", required=True)
@click.option("--output", required=True)
@click.option("--low", default=0.02, type=float)
@click.option("--high", default=0.98, type=float)
def up_quality(input, output, low, high):
    
    print(f"Оценка качества: {input} -> {output}")
    
    records = []
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    density = info_density(list(rec_entropy(records)))
    print(f"Информационная плотность: {density:.4f} nats/token")
    
    unique = list(remove_dupl(records))
    print(f"Дубликатов удалено: {len(records) - len(unique)}")
    
    filtered = filter_entropy(unique, low, high)
    print(f"Удалено по энтропии: {len(unique) - len(filtered)}")
    
    with open(output, "w", encoding="utf-8") as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Сохранено записей: {len(filtered)}")

@cli.command()
@click.option("--input", required=True)
@click.option("--tok-type", default="bpe", type=click.Choice(["char", "word", "bpe"]))
@click.option("--vocab-size", default=2000,  type=int)
def tokenization(input, tok_type, vocab_size):
    output = os.path.join(DATA_DIR, f"tokenization/{tok_type}_vocab.json")
    tokenizer(input, output, tok_type, vocab_size)

@cli.command()
@click.option("--input", required=True)
@click.option("--output", required=True)
@click.option("--max-length", default=512)
@click.option("--pad-id", default=0)
def pack(input, output, max_length, pad_id):
    # Загрузка словаря
    with open('data/tokenization/bpe_vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    tokenized_file = 'data/tokenized_sequences.jsonl'
    
    with open(input, 'r', encoding='utf-8') as f_in, \
         open(tokenized_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if line.strip():
                data = json.loads(line)
                text = data.get('text', '')
                # Токенизация текста в числовые ID
                tokens = [vocab.get(char, 0) for char in text]
                f_out.write(json.dumps({'tokens': tokens}) + '\n')
    
    print(f"Создан файл с токенами: {tokenized_file}")
    
    # Упаковывкаа токенов
    seqs = [json.loads(line)['tokens'] for line in open(tokenized_file) if line.strip()]
    print(f"Упаковка {len(seqs)} последовательностей...")
    ids, mask = packing(seqs, max_length, pad_id)
    np.savez_compressed(output, input_ids=ids, attention_mask=mask)
    print(f"Создано {len(ids)} батчей")

if __name__ == "__main__":
    cli()

import json
from pathlib import Path
from collections import Counter, defaultdict

def char_tokenizer(texts, vocab_size):
    """Возвращает список уникальных символов"""
    chars = set()
    for text in texts:
        chars.update(text)
    chars = sorted(chars)[:vocab_size] 
    print(f"✓ Сформирован словарь в {len(chars)} токенов")
    return {token: i for i, token in enumerate(chars)}

def word_tokenizer(texts, vocab_size):
    """Возвращает список уникальных слов (ограниченный vocab_size)"""
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    words = [word for word, _ in word_counts.most_common(vocab_size)]
    print(f"✓ Сформирован словарь в {len(words)} токенов")
    return {token: i for i, token in enumerate(words)}

def bpe_tokenizer(texts, vocab_size):
    print(f"Обучение BPE на {len(texts)} текстах...")
    if not texts:
        print("BPE: Нет текстов!")
        return {}

    # Начинаем с символов
    all_text = " ".join(texts)
    tokens = list(all_text)
    
    print(f"Начальное количество символов: {len(set(tokens))}")
    
    # Итеративно объединяем частые пары
    for step in range(vocab_size - len(set(tokens))):
        # Считаем пары
        pairs = defaultdict(int)
        for i in range(len(tokens)-1):
            pairs[(tokens[i], tokens[i+1])] += 1
        
        if not pairs:
            break
            
        # Поиск самой частой пары
        best_pair = max(pairs, key=pairs.get)
        
        # Объединение
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
        
        if (step + 1) % 100 == 0:
            print(f"  Шаг {step+1}: {len(set(tokens))} уникальных токенов")
    
    unique_tokens = sorted(set(tokens))[:vocab_size]
    print(f"Словарь: {len(unique_tokens)} токенов")
    
    return {token: i for i, token in enumerate(unique_tokens)}

def save(vocab, path, name):
    """Сохраняет словарь в JSON файл"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Cловарь сохранён: {path}")

def tokenizer(input, output, tok_type, vocab_size):
    """Подбирает метод токенизации"""
    print(f"Чтение файла: {input}")
    
    texts = []
    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if "text" in data and data["text"].strip():
                        texts.append(data["text"])
                except Exception as e:
                    print(f"Ошибка в строке: {e}")
                    continue
    
    print(f"Загружено текстов: {len(texts)}")
    
    if not texts:
        print("ОШИБКА: Нет текстов для токенизации!")
        return
    
    tokens = None
    if tok_type == "char":
        print("Используем char токенизацию")
        tokens = char_tokenizer(texts, vocab_size)
    elif tok_type == "word":
        print("Используем word токенизацию")
        tokens = word_tokenizer(texts, vocab_size)
    elif tok_type == "bpe":
        print("Используем bpe токенизацию")
        tokens = bpe_tokenizer(texts, vocab_size)
    
    if tokens is None:
        print("ОШИБКА: Токенизация не удалась!")
        return
    
    print(f"Создано {len(tokens)} токенов")
    
    # Сохраняем
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)
    
    print(f"Словарь сохранён: {output}")

def encode(text, vocab):
    """Превращает текст в токены"""
    tokens = []
    i = 0
    while i < len(text):
        found = False
        for token in sorted(vocab.keys(), key=len, reverse=True):
            if text[i:].startswith(token):
                tokens.append(vocab[token])
                i += len(token)
                found = True
                break
        if not found:
            i += 1
    return tokens

def decode(tokens, vocab):
    """Превращает токены обратно в текст"""
    reverse_vocab = {v: k for k, v in vocab.items()}
    return ''.join(reverse_vocab.get(t, '') for t in tokens)


        
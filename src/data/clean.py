"""
Модуль для очистки данных
"""

import re
import unicodedata
from langdetect import detect, DetectorFactory

SEED = 239
DetectorFactory.seed = SEED


def remove_wet_metadata(text):
    """Удаляет метаданные WET файла"""
    if '\n\n' in text:
        return text.split('\n\n', 1)[1]
    return text


def normalize_whitespace(text):
    """Нормализует пробелы"""
    return ' '.join(text.split())


def normalize_unicode(text):
    """Нормализует Unicode символы"""
    return unicodedata.normalize('NFKC', text)


def filter_by_language(text, min_length=20):
    """Оставляет только русский и английский язык"""
    sentences = re.split(r'[.!?]+', text)
    filtered = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < min_length:
            continue
        try:
            lang = detect(sentence)
            if lang in ['ru', 'en']:
                filtered.append(sentence)
        except:
            if any(c.isalpha() for c in sentence):
                filtered.append(sentence)
    
    return '. '.join(filtered)


def remove_bad_chars(text):
    """Удаляет нежелательные символы"""
    pattern = r'[^\w\s\.\,\!\?\-:;\(\)\'\"\[\]\{\}@#\$%&*+=/\\|<>]'
    return re.sub(pattern, '', text, flags=re.UNICODE)


def fragmentation(sequences, chunk_size=512):
    """
    Разбивает последовательности токенов на фрагменты равной длины
    
    Args:
        sequences: список списков токенов
        chunk_size: целевая длина фрагмента
    
    Returns:
        dict: {"InpData": список фрагментов}
    """
    inp_flow = []
    for seq in sequences:
        inp_flow.extend(seq)
    
    length = (len(inp_flow) // chunk_size) * chunk_size
    chunks = [inp_flow[i:i+chunk_size] for i in range(0, length, chunk_size)]
    
    return {"InpData": chunks}


def clean_wet_data(text, use_lang_filter=False):
    """
    Полная очистка WET данных
    
    Args:
        text: исходный текст
        use_lang_filter: использовать ли фильтрацию по языку (медленно)
    """
    text = remove_wet_metadata(text)
    text = normalize_whitespace(text)
    text = normalize_unicode(text)
    
    if use_lang_filter:
        text = filter_by_language(text)
    
    text = remove_bad_chars(text)
    text = normalize_whitespace(text)
    
    return text

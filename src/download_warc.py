"""Функции для загрузки WARC-файла из Common Crawl"""

import gzip
import io
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_PATHS_URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2014-15/warc.paths.gz"
DEFAULT_TIMEOUT = 180
KILOBYTE = 1024

def find_smallest_warc_url(paths_url: str = DEFAULT_PATHS_URL) -> str:

    print(f"Загрузка paths.gz: {paths_url}")
    http_response = requests.get(paths_url, timeout=DEFAULT_TIMEOUT)
    http_response.raise_for_status()

    with gzip.GzipFile(fileobj=io.BytesIO(http_response.content)) as gz_file:
        warc_paths = [line.decode('utf-8').strip() for line in gz_file if line.strip()]
    
    if not warc_paths:
        raise ValueError(f"Файл {paths_url} не содержит WARC-путей")
    
    print(f"Найдено {len(warc_paths)} WARC-файлов. Определяем размеры...")
    
    smallest_url = None
    smallest_size = float('inf')
    
    for i, path in enumerate(warc_paths[:10]):
        warc_url = "/".join(paths_url.split("/")[:3]) + "/" + path
        try:
            head_response = requests.head(warc_url, timeout=DEFAULT_TIMEOUT)
            head_response.raise_for_status()
            size = int(head_response.headers.get('content-length', 0))
            
            if size > 0 and size < smallest_size:
                smallest_size = size
                smallest_url = warc_url
                
            if (i + 1) % 100 == 0:
                print(f"  Проверено {i + 1} файлов...")
                
        except requests.RequestException as e:
            print(f"  Предупреждение: не удалось получить размер {warc_url}: {e}")
            continue
    
    if smallest_url is None:
        raise ValueError("Не удалось найти ни одного доступного WARC-файла")
    
    print(f"Наименьший файл в манифесте: {smallest_url}")
    print(f"Размер: {smallest_size / 1024 / 1024:.2f} MB")
    return smallest_url

def get_warc_url(paths_url: str, i: int = 0) -> str:
    """Возвращает URL (i-го WARC-файла из манифеста."""
    print(f"Загрузка манифеста: {paths_url}")
    http_response = requests.get(paths_url, timeout=DEFAULT_TIMEOUT)
    http_response.raise_for_status()

    with gzip.GzipFile(fileobj=io.BytesIO(http_response.content)) as gz_file:
        for current_index, line in enumerate(gz_file):
            if line.strip():
                if current_index == i:
                    selected_path = line.decode('utf-8').strip()
    
    return "/".join(paths_url.split("/")[:3]) + "/" + selected_path

def download_warc(
        paths_url: str = DEFAULT_PATHS_URL, 
        output_path: str = DATA_DIR / "RawWarc.gz", 
        chunk_size: int = KILOBYTE * KILOBYTE,
        smallest: bool = False,
        warc_index: int = 0, 
        ) -> Path:

    if smallest:
        url = find_smallest_warc_url(paths_url)
    else:
        url = get_warc_url(paths_url, warc_index)
    output_path = Path(output_path) 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Скачивание warc: {url}")
    with requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        
        with output_path.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
    
    print(f"Архив WARC.gz успешно сохранен на диск ✓")
    return output_path

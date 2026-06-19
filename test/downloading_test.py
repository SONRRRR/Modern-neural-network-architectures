"""Тестирование скачивания WARC-файлов"""

import sys
from pathlib import Path

# Добавляем КОРЕНЬ проекта в путь (где папки src/, test/, data/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Теперь импорт работает от корня проекта
from src.data.download_warc import download_warc


def test_download_smallest():
    """Тест: скачать самый маленький WARC"""
    print("=" * 60)
    print("Тест: Скачивание самого маленького WARC-файла")
    print("=" * 60)
    
    try:
        result = download_warc(
            paths_url="https://data.commoncrawl.org/crawl-data/CC-MAIN-2014-15/warc.paths.gz",
            smallest=True,
            output_path=None
        )
        print(f"\n✅ Файл сохранён: {result}")
        
        size_mb = result.stat().st_size / 1024 / 1024
        print(f"📦 Размер: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


def test_connection():
    """Тест: проверка соединения с Common Crawl"""
    import requests
    
    url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2014-15/warc.paths.gz"
    print(f"Проверка соединения с {url}")
    
    try:
        response = requests.head(url, timeout=30)
        print(f"Статус: {response.status_code}")
        print("✅ Соединение работает!")
        return True
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False


if __name__ == "__main__":
    # Сначала проверяем соединение
    if test_connection():
        # Если соединение есть — скачиваем самый маленький WARC
        test_download_smallest()
    else:
        print("\n💡 Нет соединения с Common Crawl.")
        print("Попробуйте:")
        print("  1. Проверить интернет")
        print("  2. Использовать VPN")
        print("  3. Попробовать позже")
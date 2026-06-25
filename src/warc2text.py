import gzip
import json
from pathlib import Path
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def html_removal(html: bytes) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text("\n")
    except Exception as e:
        # Если не удалось распарсить, возвращаем пустую строку
        return ""

def conversion(warc_path, lim: int = 0):
    with gzip.open(warc_path, 'rb') as stream:
        total_records, response_records,html_records = 0, 0, 0
        
        for i, record in enumerate(ArchiveIterator(stream)):
            total_records += 1            
            if lim and i >= lim:
                break
            if record.rec_type != "response":
                continue
            response_records += 1
            content = (record.http_headers.get_header("Content-Type") or "") if record.http_headers else ""
            if "html" not in content.lower():
                continue
            html_records += 1
            payload = record.content_stream().read()
            if not payload:
                continue
            yield {
                "id": record.rec_headers.get_header("WARC-Record-ID"),
                "url": record.rec_headers.get_header("WARC-Target-URI"),
                "text": html_removal(payload)
            }
    print(f"Всего записей={total_records}, response={response_records}, html={html_records}")

def warc2jsonl(warc_path, out_path, lim: int = None):
    if out_path is None:
        out_path = DATA_DIR / f"{Path(warc_path).stem}.jsonl"
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for record in conversion(warc_path, lim):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
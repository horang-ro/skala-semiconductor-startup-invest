"""Markdown 보고서를 PDF로 변환합니다 (HTML → 헤드리스 Chrome 인쇄)."""

import shutil
import subprocess
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif;
       font-size: 10pt; line-height: 1.55; color: #111; }
h1 { font-size: 16pt; border-bottom: 2px solid #333; padding-bottom: 3px; }
h2 { font-size: 13pt; margin-top: 16px; }
h3 { font-size: 11pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 6px 0; }
td, th { border: 1px solid #999; padding: 3px 5px; vertical-align: top; }
th { background: #f0f0f0; }
a { color: #1a56b0; word-break: break-all; }
"""

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
]


def find_chrome() -> str | None:
    """시스템에 설치된 Chrome/Chromium 실행 파일을 찾습니다."""
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if path := shutil.which(name):
            return path
    return next((p for p in CHROME_CANDIDATES if Path(p).exists()), None)


def markdown_to_pdf(md_text: str, pdf_path: Path) -> Path:
    """Markdown을 HTML로 바꾼 뒤 PDF로 저장합니다. Chrome이 없으면 HTML만 저장합니다."""
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )

    chrome = find_chrome()
    if chrome is None:
        print(f"Chrome을 찾지 못해 HTML만 저장했습니다: {html_path}")
        return html_path

    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path.resolve()}", html_path.resolve().as_uri()],
        check=True, capture_output=True, timeout=120,
    )
    return pdf_path

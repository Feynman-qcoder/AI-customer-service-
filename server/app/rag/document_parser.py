from io import BytesIO
from typing import Any

from app.core.exceptions import AppError

TEXT_EXTENSIONS = {"txt", "md", "markdown", "csv", "json", "log"}


def extract_text(filename: str, content: bytes) -> str:
    extension = _extension(filename)
    if extension in TEXT_EXTENSIONS:
        return _decode_text(content)
    if extension == "pdf":
        return _extract_pdf(content)
    if extension == "docx":
        return _extract_docx(content)
    raise AppError("仅支持 txt、md、markdown、csv、json、pdf、docx 文档", 400)


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _extract_pdf(content: bytes) -> str:
    try:
        from importlib import import_module

        pdf_module: Any = import_module("pypdf")
        reader = pdf_module.PdfReader(BytesIO(content))
    except Exception as exc:
        raise AppError("当前环境未安装 pypdf，暂不能解析 PDF", 500) from exc
    return "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()


def _extract_docx(content: bytes) -> str:
    try:
        from importlib import import_module

        docx_module: Any = import_module("docx")
        document = docx_module.Document(BytesIO(content))
    except Exception as exc:
        raise AppError("当前环境未安装 python-docx，暂不能解析 DOCX", 500) from exc
    return "\n".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip())

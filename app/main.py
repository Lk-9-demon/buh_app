from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.dependencies import (
    get_chat_service,
    get_file_registry,
    get_indexing_manager,
    get_ollama_service,
    get_search_index,
)
from app.services.document_preview import (
    build_preview_context,
    guess_media_type,
    render_answer_html,
    resolve_document_path,
)
from app.types import ChatAnswer


settings = get_settings()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_directories()
    get_file_registry()
    get_search_index().ensure_collections()
    yield


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, message: str | None = None):
    return _render_home(request=request, message=message)


@app.get("/documents/file")
async def serve_document_file(path: str):
    try:
        resolved_path = resolve_document_path(settings, path)
    except (FileNotFoundError, PermissionError, ValueError):
        raise HTTPException(status_code=404, detail="Document not found")

    return FileResponse(path=resolved_path, media_type=guess_media_type(resolved_path), filename=resolved_path.name)


@app.get("/documents/preview", response_class=HTMLResponse)
async def preview_document(
    request: Request,
    path: str,
    page: int | None = None,
    sheet_name: str | None = None,
    row_number: int | None = None,
):
    try:
        resolved_path = resolve_document_path(settings, path)
    except (FileNotFoundError, PermissionError, ValueError):
        raise HTTPException(status_code=404, detail="Document not found")

    preview = await run_in_threadpool(
        build_preview_context,
        resolved_path,
        page,
        sheet_name,
        row_number,
    )
    return templates.TemplateResponse(
        request=request,
        name="preview.html",
        context={
            "app_name": settings.app_name,
            "preview": preview,
        },
    )


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] | None = File(default=None)):
    if not files:
        return _redirect_with_message("Не вибрано жодного файла для завантаження.")

    saved = 0
    skipped: list[str] = []
    for upload in files:
        filename = Path(upload.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in settings.allowed_extensions:
            skipped.append(filename or "unknown")
            await upload.close()
            continue

        target_path = _next_available_path(settings.documents_dir, filename)
        bytes_written = 0
        with target_path.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_size_mb * 1024 * 1024:
                    handle.close()
                    target_path.unlink(missing_ok=True)
                    await upload.close()
                    return _redirect_with_message(
                        f"Файл {filename} перевищує ліміт {settings.max_upload_size_mb} MB."
                    )
                handle.write(chunk)
        await upload.close()
        saved += 1

    message = f"Завантажено {saved} файл(и)."
    if skipped:
        message += f" Пропущено: {', '.join(skipped)}."
    return _redirect_with_message(message)


@app.post("/documents/reindex")
async def reindex_documents():
    runtime_status = get_ollama_service().get_status()
    if not runtime_status["ready"]:
        return _redirect_with_message(str(runtime_status["message"]))

    started = get_indexing_manager().start()
    if not started:
        return _redirect_with_message("Індексація вже триває. Онови сторінку, щоб побачити прогрес.")

    return _redirect_with_message("Індексацію запущено у фоні. Сторінка більше не повинна зависати.")


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, question: str = Form(...)):
    runtime_status = get_ollama_service().get_status()
    if not runtime_status["ready"]:
        return _render_home(
            request=request,
            chat_answer=ChatAnswer(
                question=question,
                answer=str(runtime_status["message"]),
                sources=[],
            ),
        )

    try:
        answer = await run_in_threadpool(get_chat_service().answer_question, question)
    except Exception as exc:
        answer = ChatAnswer(
            question=question,
            answer=f"Не вдалося обробити питання: {exc}",
            sources=[],
        )
    return _render_home(request=request, chat_answer=answer)


@app.get("/health")
async def health():
    runtime_status = get_ollama_service().get_status()
    return {
        "status": "ok",
        "app": settings.app_name,
        "runtime_ready": bool(runtime_status["ready"]),
    }


def _render_home(
    request: Request,
    message: str | None = None,
    chat_answer: ChatAnswer | None = None,
):
    registry = get_file_registry()
    indexing_status = get_indexing_manager().get_status()
    runtime_status = get_ollama_service().get_status()
    records = registry.list_records(limit=200)
    stats = registry.get_stats()
    chat_answer_html = render_answer_html(chat_answer.answer, chat_answer.sources) if chat_answer else None

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "message": message,
            "settings": settings,
            "runtime_ready": bool(runtime_status["ready"]),
            "runtime_message": runtime_status["message"],
            "indexing_status": indexing_status,
            "records": records,
            "stats": stats,
            "chat_answer": chat_answer,
            "chat_answer_html": chat_answer_html,
        },
    )


def _redirect_with_message(message: str) -> RedirectResponse:
    return RedirectResponse(url=f"/?message={quote_plus(message)}", status_code=303)


def _next_available_path(directory: Path, filename: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not safe_name:
        safe_name = "uploaded_file"

    candidate = directory / safe_name
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1

    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1

    return candidate

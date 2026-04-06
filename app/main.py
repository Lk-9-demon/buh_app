from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.dependencies import get_chat_service, get_file_registry, get_ollama_service, get_search_index
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

    report = await run_in_threadpool(get_search_index().index_documents)
    message = (
        f"Індексація завершена. Нових/оновлених: {report.indexed}, "
        f"пропущено: {report.skipped}, видалено з індексу: {report.removed}, помилок: {report.failed}."
    )
    if report.errors:
        message += " Деталі нижче у таблиці файлів."
    return _redirect_with_message(message)


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
    runtime_status = get_ollama_service().get_status()
    records = registry.list_records(limit=200)
    stats = registry.get_stats()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "message": message,
            "settings": settings,
            "runtime_ready": bool(runtime_status["ready"]),
            "runtime_message": runtime_status["message"],
            "records": records,
            "stats": stats,
            "chat_answer": chat_answer,
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

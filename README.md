# Buh Chat

Веб-MVP для пошуку по внутрішніх документах компанії.

## Що вже вміє

- приймати `PDF`, `XLSX`, `DOCX`, `CSV`, `TXT`
- зберігати файли локально в `data/documents`
- будувати локальний індекс у `ChromaDB`
- вести реєстр індексації в `SQLite`
- відповідати на питання через `OpenAI`
- показувати знайдені джерела поруч із відповіддю

## Архітектура

- `FastAPI` для веб-інтерфейсу
- `ChromaDB` для векторного пошуку
- `OpenAI embeddings` для перетворення chunks та запитів у вектори
- `OpenAI chat model` для фінальної відповіді
- `pypdf`, `openpyxl`, `python-docx` для читання документів

## Структура даних

- `doc_chunks_*` collection:
  - chunks із `PDF`, `DOCX`, `TXT`, `CSV`
- `excel_rows_*` collection:
  - рядки з `XLSX`

Кожен запис у Chroma містить:

- `document`: текст chunk-а
- `embedding`: вектор від `text-embedding-3-small`
- `metadata`:
  - `source_path`
  - `source_name`
  - `file_type`
  - `doc_sha256`
  - `modified_at`
  - `page`
  - `sheet_name`
  - `row_number`
  - `parser_version`

## Запуск

1. Створи `.env` на основі `.env.example`
2. Встанови залежності:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

3. Запусти застосунок:

```powershell
.\.venv\Scripts\python.exe main.py
```

4. Відкрий:

```text
http://127.0.0.1:8000
```

## Як працювати

1. Завантаж файли через веб або поклади їх у `data/documents`
2. Натисни `Переіндексувати документи`
3. Постав питання в чаті

## Важливі обмеження MVP

- скановані `PDF` без OCR не будуть читатися коректно
- `.doc` поки не підтримується, тільки `.docx`
- `OpenAI API` все ще отримує chunks і запити, тобто це не повністю локальний режим
- для великих Excel-файлів логіку парсингу листів ще можна посилити окремими правилами

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.services.chat_service import ChatService
from app.services.openai_service import OpenAIService
from app.services.parsers import DocumentParser
from app.services.registry import FileRegistry
from app.services.search_index import SearchIndex


@lru_cache
def get_file_registry() -> FileRegistry:
    settings = get_settings()
    return FileRegistry(settings.state_db_path)


@lru_cache
def get_document_parser() -> DocumentParser:
    return DocumentParser(get_settings())


@lru_cache
def get_openai_service() -> OpenAIService:
    return OpenAIService(get_settings())


@lru_cache
def get_search_index() -> SearchIndex:
    settings: Settings = get_settings()
    return SearchIndex(
        settings=settings,
        parser=get_document_parser(),
        registry=get_file_registry(),
        openai_service=get_openai_service(),
    )


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(
        search_index=get_search_index(),
        openai_service=get_openai_service(),
    )

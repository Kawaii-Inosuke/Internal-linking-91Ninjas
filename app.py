"""FastAPI web UI for the Internal Linking Assistant (TRD §9 v2, pulled forward).

A single page at ``/`` posts to ``/suggest``. The routes are thin wrappers: they
take form input, call :func:`linker.matcher.suggest_with_config`, and hand the
result to a Jinja template. All matching logic lives in ``linker.matcher`` — the
web layer holds none of it.

Run with ``python cli.py serve`` or ``uvicorn app:app`` (see the README).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from linker import matcher
from linker.config import Config

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="Internal Linking Assistant")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the empty suggest form."""
    return templates.TemplateResponse(
        request, "index.html", {"result": None, "form": {}}
    )


@app.post("/suggest", response_class=HTMLResponse)
def suggest(
    request: Request,
    client: str = Form(...),
    keyword: str = Form(...),
    post_text: str = Form(...),
    url: str = Form(""),
) -> HTMLResponse:
    """Run the exact-keyword pass for the submitted form and render the results.

    Thin wrapper: the matching itself is entirely in
    :func:`linker.matcher.suggest_with_config`.
    """
    config = Config.from_env()
    result = matcher.suggest_with_config(
        config,
        client=client,
        keyword=keyword,
        post_text=post_text,
        current_url=url or None,
    )
    form = {"client": client, "keyword": keyword, "url": url, "post_text": post_text}
    return templates.TemplateResponse(
        request, "index.html", {"result": result, "form": form}
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe (TRD §9)."""
    return {"status": "ok"}

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

> Always use `uv` for all dependency management and running Python. Never use `pip`, `pip3`, or `python` directly.

**Install dependencies:**
```bash
uv sync
```

**Add a dependency:**
```bash
uv add <package>
```

**Remove a dependency:**
```bash
uv remove <package>
```

**Run the application:**
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app serves at `http://localhost:8000` (UI) and `http://localhost:8000/docs` (API docs).

**Environment setup:** Create a `.env` file in the project root with the following variables:
- `ANTHROPIC_API_KEY=<key>` (required for Claude provider)
- `GEMINI_API_KEY=<key>` (required for Gemini provider)  
- `LLM_PROVIDER=claude|gemini` (optional, defaults to "claude")

This is loaded by `backend/config.py` via `python-dotenv`.

## Architecture

This is a RAG chatbot that answers questions about course documents using semantic search + LLM (Claude or Gemini).

### Request Flow

1. Frontend (`frontend/script.js`) POSTs to `/api/query` with a query string and session ID
2. `backend/app.py` hands the request to `RAGSystem.query()`
3. `RAGSystem` (`rag_system.py`) passes the query to `AIGenerator.generate_response()`
4. The LLM receives the query along with a `CourseSearchTool` definition (`search_tools.py`)
5. If the LLM calls the tool, `VectorStore.search()` runs a semantic similarity query against ChromaDB
6. Search results are returned to the LLM, which generates the final response
7. `SessionManager` stores up to 2 exchanges for multi-turn context

### Document Ingestion

On startup, `RAGSystem` loads all `docs/course*.txt` files via `DocumentProcessor`:
- Parses course metadata (title, link, instructor) from the first 3 lines
- Splits content by `Lesson N: Title` markers
- Chunks lesson text into ~800-char sentence-based chunks with 100-char overlap
- Stores chunks + embeddings in ChromaDB (`./chroma_db/`)

`VectorStore` maintains two ChromaDB collections: `course_catalog` (course metadata) and `course_content` (text chunks with embeddings).

### Key Configuration (`backend/config.py`)

| Setting | Value |
|---|---|
| LLM Provider | `claude` (default) or `gemini` |
| Claude model | `claude-sonnet-4-20250514` |
| Gemini model | `gemini-1.5-flash` |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) |
| Chunk size / overlap | 800 / 100 chars |
| Max search results | 5 |
| Max conversation history | 2 exchanges |
| ChromaDB path | `./chroma_db` (relative to `backend/`) |

### Document Format

Course `.txt` files must follow this structure:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: Introduction
Lesson Link: <url>
<content>

Lesson 1: Next Topic
<content>
```

#!/usr/bin/env python3
"""Wrapper script for build_paper_kg.py with explicit stage logging.

This script imports the existing helpers from build_paper_kg.py so that we
can emit deterministic progress messages and capture any exception trace.
"""

import asyncio
import os
import sys
import traceback
from datetime import datetime
from typing import Optional

from build_paper_kg import (
    EMBED_MODEL,
    OUTPUT_DIR,
    PAPER_DIR,
    WORKING_DIR,
    build_embedding_func,
    build_llm_funcs,
    RAGAnything,
    RAGAnythingConfig,
)


def _log(stage: str, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{stage}] {message}", flush=True)


async def monitored_ingest(question: Optional[str] = None) -> None:
    _log("setup", f"Ensuring working_dir={WORKING_DIR} and output_dir={OUTPUT_DIR} exist")
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _log("embedding", f"Loading embedding model at {EMBED_MODEL}")
    embedding_func = build_embedding_func()
    _log("embedding", "Embedding function ready")

    _log("llm", "Initializing LLM and vision model functions")
    llm_model_func, vision_model_func = build_llm_funcs()
    _log("llm", "LLM/vision functions ready")

    _log("config", "Creating RAGAnythingConfig with multimodal processing enabled")
    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    _log("rag", "Instantiating RAGAnything pipeline")
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    _log("rag", "RAGAnything pipeline ready")

    _log(
        "process",
        f"Processing folder {PAPER_DIR} -> {OUTPUT_DIR} (method={config.parse_method})",
    )
    try:
        await rag.process_folder_complete(
            folder_path=PAPER_DIR,
            output_dir=OUTPUT_DIR,
            parse_method=config.parse_method,
            recursive=False,
        )
        _log("process", "Folder processing completed successfully")
    except Exception as exc:  # noqa: BLE001
        _log("process", f"Processing failed: {exc}")
        _log("process", traceback.format_exc())
        raise

    if question:
        _log("qa", f"Running sample hybrid query: {question}")
        try:
            answer = await rag.aquery(question, mode="hybrid")
            _log("qa", "Query result:")
            print(answer)
        except Exception as exc:  # noqa: BLE001
            _log("qa", f"Sample QA failed: {exc}")
            _log("qa", traceback.format_exc())
            raise
    else:
        _log("qa", "No sample question provided; skipping QA step")


def main() -> None:
    question = os.getenv(
        "RAG_SAMPLE_QUESTION",
        "Summarize the key contributions of each paper in this collection.",
    )
    _log("main", "Starting monitored ingestion pipeline")
    _log("main", f"Sample question: {question}")
    try:
        asyncio.run(monitored_ingest(question=question))
    except KeyboardInterrupt:  # noqa: PIE786
        _log("main", "Interrupted by user")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        _log("main", f"Pipeline failed: {exc}")
        sys.exit(1)
    else:
        _log("main", "Pipeline finished successfully")


if __name__ == "__main__":
    main()

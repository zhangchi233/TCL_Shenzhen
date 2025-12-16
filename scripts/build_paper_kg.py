import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

load_dotenv(dotenv_path=".env", override=False)

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3-VL-thinking")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8848/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "DUMMY_KEY")

WORKING_DIR = os.getenv("RAG_WORKING_DIR", str(Path("./rag_storage_papers").resolve()))
OUTPUT_DIR = os.getenv("RAG_OUTPUT_DIR", str(Path("./rag_output").resolve()))
PAPER_DIR = os.getenv("PAPER_DIR", "/mnt/storage/MLLM/zyf/paper")

EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "/mnt/storage/models/BAAI/bge-m3"
)

def build_embedding_func():
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    async def encode(texts):
        loop = asyncio.get_running_loop()

        def _encode_sync():
            vectors = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vectors.tolist()

        return await loop.run_in_executor(None, _encode_sync)

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=8192,
        func=encode,
    )


def build_llm_funcs():
    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        history_messages = history_messages or []
        return openai_complete_if_cache(
            LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        history_messages = history_messages or []
        if messages:
            return openai_complete_if_cache(
                LLM_MODEL,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                **kwargs,
            )
        if image_data:
            payload = [
                {"role": "system", "content": system_prompt}
                if system_prompt
                else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                },
            ]
            payload = [m for m in payload if m]
            return openai_complete_if_cache(
                LLM_MODEL,
                "",
                system_prompt=None,
                history_messages=[],
                messages=payload,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                **kwargs,
            )
        return llm_model_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    return llm_model_func, vision_model_func


async def ingest_papers(question: str | None = None):
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    embedding_func = build_embedding_func()
    llm_model_func, vision_model_func = build_llm_funcs()

    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    await rag.process_folder_complete(
        folder_path=PAPER_DIR,
        output_dir=OUTPUT_DIR,
        parse_method=config.parse_method,
        recursive=False,
    )

    if question:
        answer = await rag.aquery(question, mode="hybrid")
        print("\n=== Sample QA Result ===")
        print(question)
        print("---")
        print(answer)
        print("=======================\n")


def main():
    question = os.getenv(
        "RAG_SAMPLE_QUESTION",
        "Summarize the key contributions of each paper in this collection.",
    )
    asyncio.run(ingest_papers(question=question))


if __name__ == "__main__":
    main()

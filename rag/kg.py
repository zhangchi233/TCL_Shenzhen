#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)
MODEL_NAME = "ep-20251124134322-8h8rn"

def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    api_key: str = "d69ffc82-6fdd-48ea-bff3-5dd4daf8439a",
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    parser: str = "mineru",
):
    """
    Process document with RAGAnything

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=file_path,
            parser=parser,  # Parser selection: mineru or docling
            parse_method="auto",  # Parse method: auto, ocr, or txt
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Define LLM model function
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                MODEL_NAME,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Define vision model function for image processing
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                return openai_complete_if_cache(
                    MODEL_NAME,
                    api_key,
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # Traditional single image format
            elif image_data:
                return openai_complete_if_cache(
                    MODEL_NAME,
                    api_key,
                    system_prompt=None,
                    history_messages=[],
                    messages=[
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
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    timeout=120,
                    **kwargs,
                )
            # Pure text format
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Define embedding function - using environment variables for configuration
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "2048"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="ep-20251216001936-l62nr",
                api_key=api_key,
                base_url="https://ark.cn-beijing.volces.com/api/v3",
            ),
        )

        # Initialize RAGAnything with new dataclass structure
        rag_light = LightRAG(
            working_dir=file_path,
            llm_model_func=llm_model_func,
            #vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )
        
        await rag_light.initialize_storages()
        
        rag = RAGAnything(lightrag=rag_light, vision_model_func=vision_model_func,config =config)
        await rag.finalize_storages()
        # img="/home/maxzhang/datapipeline/temp_images/img_in_chart_box_195_1973_1126_2427.jpg"
        # question = "OLED 相关的专利申请情况有什么趋势？"
        # print("IMAGE_PATH:", img, "\n Query: ", question)

        # multimodal_result = await rag.aquery_with_multimodal(
        #     question,
        #     multimodal_content=[
        #         {
        #             "type": "image",
        #             "image_path": img,
        #         }
        #     ],
        #     mode="hybrid",
        # )
        # # await rag.initialize_storages()
        # print(multimodal_result)
        return rag
        # img="/mnt/storage/dataset/PPVL_reuslts_CN/中文/wanfang202507/微显示技术/微显示技术/硅基OLED/一种提高硅基OLED微显示器温补能力的驱动电路设计与实现/imgs/img_in_image_box_643_1646_1738_2411.jpg"
        # question = "What does this image show?"
        # print("IMAGE_PATH:", img, "\n Query: ", question)

        # multimodal_result = await rag.aquery_with_multimodal(
        #     "Describe this image",
        #     multimodal_content=[
        #         {
        #             "type": question,
        #             "image_path": img,
        #         }
        #     ],
        #     mode="hybrid",
        # )

        # print(multimodal_result)



    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
async def run_single_test(rag,img = None,caption="玻璃刻蚀速率随HF浓度升高而增大，且增大速率随浓度升高而加快，呈非线性关系。"):
    """
    运行单次测试，验证 API 是否正常工作
    """
    # print("--- 运行单次测试 ---")
    
    query = caption    
    
    query= query+"请返回和上述内容相关的背景知识。"
    try:
        res = await rag.aquery_with_multimodal(
                query,
                multimodal_content=[
                    {
                        "type": "image",
                        "image_path": img,
                    }
                ],
                mode="hybrid",
            )
        
        print(f"✅ 单次调用成功，结果: {res}")
        return res
        
            
       
    except Exception as e:
        print(f"❌ 单次调用失败: {e}")
        return "未找到相关背景知识。"

def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument("--file_path",default = "/mnt/storage/dataset/PPVL_reuslts_CN/RAG-Anything/pdfs/氧化物TFT源栅极短路缺陷原因解析及抑制措施_NormalPdf.pdf", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="/mnt/storage/dataset/PPVL_reuslts_CN/RAG-Anything/examples/local/pdfs/rag", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="/mnt/storage/dataset/PPVL_reuslts_CN/RAG-Anything/examples/local/pdfs/output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default="d69ffc82-6fdd-48ea-bff3-5dd4daf8439a",
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default="https://ark.cn-beijing.volces.com/api/v3",
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # Check if API key is provided
    # if not args.api_key:
    #     logger.error("Error: OpenAI API key is required")
    #     logger.error("Set api key environment variable or use --api-key option")
    #     return

    # Create output directory if specified
   
    # Process with RAG
    asyncio.run(
        process_with_rag(
        file_path= "/home/maxzhang/RAG-Anything/rag_test_examples",
        api_key= "d69ffc82-6fdd-48ea-bff3-5dd4daf8439a",
        base_url = "https://ark.cn-beijing.volces.com/api/v3",
        parser = "mineru",
    )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()

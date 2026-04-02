import asyncio
import os
from random import sample
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from rich.panel import Panel
from rich.console import Console
from urllib3.util import url

from logger import (Colors, log_info, log_warning, log_error, log_success, log_header)

load_dotenv()
console = Console()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10
)

# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings,
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(
    max_depth=5, 
    max_breadth=50, 
    max_pages=1000,
    allowed_domains=[
        "docs.langchain.com",
        # "changelog.langchain.com",
        # "blog.langchain.dev",
        ]
)
tavily_crawl = TavilyCrawl()


# Forma avançada
async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(" TavilyMap: Starting to map documentation structure from https://docs.langchain.com/", Colors.PURPLE)

    site_map = tavily_map.invoke({
        "url": "https://docs.langchain.com/"
    })
    # site_map = tavily_map.invoke("https://docs.langchain.com/")
    log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site")
    
    # Split URLs into batches of 20
    url_batches = chunk_urls(list(site_map['results']), chunk_size=20)
    log_info(f" URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches", Colors.BLUE)

    # Extract documents from URLs
    all_docs = await async_extract(url_batches)
    
    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f" Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap", Colors.YELLOW)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents")
    #  Continuar com a inserção dos dados da vindo da web , para a base de dados vetorial, agora pegar as partes e converter para vetores

# Forma normal
# async def main():
#     """Main async function to orchestrate the entire process."""

#     log_header("DOCUMENTATION INGESTION PIPELINE")
#     log_info(
#         "TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/",
#         Colors.PURPLE
#         )

#     # Crawl the documentation site
#     res = tavily_crawl.invoke({
#         "url": "https://langchain.com/",
#         "max_depth": 1,
#         # "max_depth": 5,
#         "extract_depth": "advanced",
#         "instructions": "Only return content from the Langchain documentation; do not return anything other than information from the Langchain documentation.",
#     })

#     all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]

#     log_success(
#         f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site",
#     )

#     # Example website to map
#     demo_url = "https://langchain.com/"

#     log_info(f"Mapping website structure for: {demo_url}", Colors.BLUE)
#     log_info("This may take a moment...", Colors.BLUE)

#     # Map the website structure
#     site_map = tavily_map.invoke(demo_url)

#     # Display results
#     urls = site_map.get("results", [])
#     print(f"Successfully mapped {len(urls)} URLs")
    
#     # Show first 10 URLs as examples
#     # log_info("First 50 discovered URLs", Colors.BLUE)
#     # for i, url in enumerate(urls[:50], 1):
#     #     log_info(f" {i:2d}. {url}")

#     # if len(urls) > 10:
#     #     print(f"... and {len(urls) - 50} more URLs")

#     # Select a few interesting URLs for extraction
#     sample_urls = [urls[20]] # Take first 5 URLs
#     print(f"Extracting content from {len(sample_urls)} URLs...")
#     # Extract content 
#     extraction_result = await tavily_extract.ainvoke(input={"urls": sample_urls})
#     # Display results
#     extracted_docs = extraction_result.get('results', [])
#     print(f"\n Successfully extracted {len(extracted_docs)} documents:")

#     # Show summary of each extracted document
#     # for i, doc in enumerate(extracted_docs, 1):
#     #     url = doc.get('url', 'Unknown')
#     #     content = doc.get('raw_content', '')

#     #     # Create a panel for each document
#     #     panel_content = f"""URL: {url}"
#     #     Content Length: {len(content):,} characters
#     #     Preview: {content}..."""

#     #     console.print(Panel(panel_content, title=f"Document {i}"))
#     #     print()

#     # Process a larger set of URLs in batches
#     url_batches = chunk_urls(urls[:9], chunk_size=3)

#     console.print(f"Processing 9 URLs in {len(url_batches)} batches", style="bold yellow")

#     # Process batches concurrently
#     tasks = [extract_batch(batch, i +1) for i, batch in enumerate(url_batches)]
#     batch_results = await asyncio.gather(*tasks)

#     # Flatten results
#     all_extracted = []
#     for batch_result in batch_results:
#         all_extracted.extend(batch_result) 

#     console.print(f"\n Batch processing complete! Total document extracted: {len(all_extracted)}", style="bold green")



def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLs chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks    

# async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
#     """Extract documents from a batch of URLs."""
#     try:
#         console.print(f"Processing batch {batch_num} with {len(urls)} URLs", style='blue')
#         docs = await tavily_extract.ainvoke(input={"urls": urls})
#         results = docs.get("results", [])
#         console.print(f"Batch {batch_num} completed - extracted {len(results)} documents", style="green")
#         return results
#     except Exception as e:
#         console.print(f"Batch {batch_num} failed: {e}", style="red")
#         return []



async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        log_info(f" TavilyExtract: Processing batch {batch_num} whit {len(urls)}", Colors.BLUE)
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []

async def async_extract(url_batches: List[List[str]]): 
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(f" TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches", Colors.DARKCYAN)

    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0

    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result['results']:
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]}
                )    
                all_pages.append(document)
    
    log_success(f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}")
    
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")

    return all_pages



if __name__ == "__main__":
    asyncio.run(main())

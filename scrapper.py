#just a refrence code on how to scrap from the website 
#!pip install scrapy scrapy-playwright beautifulsoup4 "unstructured[pdf]" chromadb sentence-transformers langchain
#the scrapper ran on colab to fetch the data 
#due to less time not much data is scrapped.

import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.settings import Settings
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy_playwright.page import PageMethod
import logging
from pathlib import Path
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import urljoin
import time
import base64
import re
from twisted.internet import reactor, defer
import threading
import itertools
import sys

# --- Data Processing Imports ---
from bs4 import BeautifulSoup
from unstructured.partition.pdf import partition_pdf
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==============================================================================
# == SPIDER DEFINITIONS ==
# ==============================================================================

class NABARDSpider(scrapy.Spider):
    """Spider specifically for the NABARD website."""
    name = 'nabard'
    allowed_domains = ['nabard.org']
    start_urls = [
        'https://www.nabard.org/content1.aspx?id=23&catid=23&mid=530',
        'https://www.nabard.org/content1.aspx?id=25&catid=23&mid=530',
    ]

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta={'playwright': True, 'playwright_page_methods': [PageMethod('wait_for_timeout', 3000)]}
            )

    def parse(self, response):
        for link in response.css('a[href*="content1.aspx"]::attr(href)').getall():
            absolute_url = urljoin(response.url, link)
            yield scrapy.Request(url=absolute_url, callback=self.parse_scheme, meta={'playwright': True})

    def parse_scheme(self, response):
        title = response.css('.middle-content h3::text').get()
        content = response.css('.middle-content').get()
        pdf_links = response.css('a[href$=".pdf"]::attr(href)').getall()
        
        yield {
            'source_url': response.url,
            'document_title': title.strip() if title else 'NABARD Scheme',
            'issuing_body': 'NABARD',
            'content_html': content,
            'pdf_links': [urljoin(response.url, link) for link in pdf_links],
            'last_scraped_ts': time.time(),
            'document_type': 'HTML'
        }
        
        for pdf_link in pdf_links:
            yield scrapy.Request(url=urljoin(response.url, pdf_link), callback=self.parse_pdf, meta={'source_url': response.url, 'document_title': title})

    def parse_pdf(self, response):
        yield {
            'source_url': response.meta['source_url'],
            'document_title': response.meta['document_title'],
            'issuing_body': 'NABARD',
            'pdf_content': base64.b64encode(response.body).decode('utf-8'),
            'pdf_url': response.url,
            'last_scraped_ts': time.time(),
            'document_type': 'PDF'
        }

class SBISpider(scrapy.Spider):
    """Spider specifically for the SBI website."""
    name = 'sbi'
    allowed_domains = ['sbi.co.in']
    start_urls = ['https://sbi.co.in/web/agri-rural']
    
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta={'playwright': True, 'playwright_page_methods': [PageMethod('wait_for_timeout', 3000)]}
            )

    def parse(self, response):
        for link in response.css('a[href*="agri-rural/"]::attr(href)').getall():
            absolute_url = urljoin(response.url, link)
            if 'javascript:void(0)' not in absolute_url:
                yield scrapy.Request(url=absolute_url, callback=self.parse_loan_product, meta={'playwright': True})

    def parse_loan_product(self, response):
        title = response.css('h1.page-title::text').get()
        content = response.css('.right-panel').get()
        
        yield {
            'source_url': response.url,
            'document_title': title.strip() if title else 'SBI Loan',
            'issuing_body': 'SBI',
            'content_html': content,
            'pdf_links': [],
            'last_scraped_ts': time.time(),
            'document_type': 'HTML'
        }

class GeneralSpider(CrawlSpider):
    """A general-purpose, configurable spider."""
    name = 'general'

    def __init__(self, *args, **kwargs: Any):
        self.issuing_body = kwargs.get('issuing_body', 'Unknown')
        self.start_urls = kwargs.get('start_urls', '').split(',')
        self.allowed_domains = kwargs.get('allowed_domains', '').split(',')
        
        allow_rules = kwargs.get('allow_rules', ())
        if isinstance(allow_rules, str):
            allow_rules = (allow_rules,)

        GeneralSpider.rules = [
            Rule(LinkExtractor(allow=allow_rules), callback='parse_page', follow=True),
        ]
        super(GeneralSpider, self).__init__(*args, **kwargs)

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse_page,
                meta={'playwright': True, 'playwright_page_methods': [PageMethod('wait_for_load_state', 'domcontentloaded')]}
            )

    def parse_page(self, response):
        title = response.css('h1::text, .page-title::text, .entry-title::text, title::text').get()
        content = response.css('article, .content, .main-content, .entry-content, body').get()
        pdf_links = response.css('a[href$=".pdf"]::attr(href)').getall()
        
        yield {
            'source_url': response.url,
            'document_title': title.strip() if title else 'No Title Found',
            'issuing_body': self.issuing_body,
            'content_html': content,
            'pdf_links': [urljoin(response.url, link) for link in pdf_links],
            'last_scraped_ts': time.time(),
            'document_type': 'HTML'
        }
        
        for pdf_link in pdf_links:
            yield scrapy.Request(url=urljoin(response.url, pdf_link), callback=self.parse_pdf, meta={'source_url': response.url, 'document_title': title})

    def parse_pdf(self, response):
        yield {
            'source_url': response.meta.get('source_url'),
            'document_title': response.meta.get('document_title'),
            'issuing_body': self.issuing_body,
            'pdf_content': base64.b64encode(response.body).decode('utf-8'),
            'pdf_url': response.url,
            'last_scraped_ts': time.time(),
            'document_type': 'PDF'
        }

# ==============================================================================
# == DATA PROCESSOR CLASS ==
# ==============================================================================

class DataProcessor:
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path=str(self.output_dir / "chroma_db"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="agricultural_loans",
            metadata={"hnsw:space": "cosine"}
        )
        self.logger = logging.getLogger(__name__)

    def process_scraped_data(self, scraped_data_file: str):
        self.logger.info(f"Starting data processing from {scraped_data_file}")
        with open(scraped_data_file, 'r', encoding='utf-8') as f:
            scraped_items = [json.loads(line) for line in f]
        
        processed_documents = []
        for item in scraped_items:
            docs = []
            if item.get('document_type') == 'PDF' and 'pdf_content' in item:
                docs = self.process_pdf_item(item)
            elif item.get('document_type') == 'HTML' and 'content_html' in item:
                docs = self.process_html_item(item)
            processed_documents.extend(docs)
        
        self.store_in_vector_db(processed_documents)
        
        output_file = self.output_dir / "processed_documents.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_documents, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"Processed and stored {len(processed_documents)} document chunks.")

    def process_pdf_item(self, item: Dict) -> List[Dict]:
        documents = []
        try:
            pdf_path = self.output_dir / f"{hashlib.md5(item['pdf_url'].encode()).hexdigest()}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(base64.b64decode(item['pdf_content']))

            elements = partition_pdf(str(pdf_path), strategy="fast")
            text_content = "\n".join([str(el) for el in elements])
            chunks = self.chunk_text(text_content)
            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': self.generate_doc_id(item['source_url'], f"pdf_chunk_{i}"),
                    'content': chunk,
                    'metadata': self.extract_metadata(item, 'PDF'),
                })
            pdf_path.unlink()
        except Exception as e:
            self.logger.error(f"Error processing PDF from {item.get('pdf_url')}: {e}")
        return documents

    def process_html_item(self, item: Dict) -> List[Dict]:
        documents = []
        try:
            soup = BeautifulSoup(item['content_html'], 'html.parser')
            text_content = self.clean_text(soup.get_text())
            chunks = self.chunk_text(text_content)
            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': self.generate_doc_id(item['source_url'], f"html_chunk_{i}"),
                    'content': chunk,
                    'metadata': self.extract_metadata(item, 'HTML'),
                })
        except Exception as e:
            self.logger.error(f"Error processing HTML from {item.get('source_url')}: {e}")
        return documents

    def extract_metadata(self, item: Dict, doc_type: str) -> Dict:
        return {
            'source_url': item.get('source_url', ''),
            'document_title': item.get('document_title', 'No Title'),
            'issuing_body': item.get('issuing_body', 'Unknown'),
            'last_scraped_ts': str(datetime.fromtimestamp(item.get('last_scraped_ts', 0))),
            'document_type': doc_type
        }

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, length_function=len
        )
        return text_splitter.split_text(text)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def generate_doc_id(self, url: str, chunk_info: Any) -> str:
        return hashlib.md5(f"{url}_{chunk_info}".encode()).hexdigest()

    def store_in_vector_db(self, documents: List[Dict]):
        if not documents:
            self.logger.warning("No documents to store.")
            return
        self.logger.info(f"Generating embeddings for {len(documents)} chunks...")
        contents = [doc['content'] for doc in documents]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
        ids = [doc['id'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        self.collection.upsert(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=contents)
        self.logger.info("Successfully stored documents in vector database.")

# ==============================================================================
# == LOADING ANIMATION ==
# ==============================================================================

def animate(stop_event):
    """A simple command-line spinner animation."""
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\rScraping in progress... {c}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rScraping complete!           \n')
    sys.stdout.flush()

# ==============================================================================
# == SCRAPER RUNNER LOGIC (Updated to use CrawlerRunner) ==
# ==============================================================================

def run_scrapers():
    """Configures and runs all scrapers, then processes the data."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    output_file = 'scraped_data.jsonl'
    
    settings = Settings({
        'BOT_NAME': 'agricultural_loan_scraper',
        'ROBOTSTXT_OBEY': False,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
        'DOWNLOAD_DELAY': 2, 'RANDOMIZE_DOWNLOAD_DELAY': True, 'AUTOTHROTTLE_ENABLED': True,
        'FEEDS': {output_file: {'format': 'jsonlines', 'encoding': 'utf8', 'overwrite': True}},
        'DOWNLOAD_HANDLERS': {"http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler", "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler"},
        'TWISTED_REACTOR': "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        'LOG_LEVEL': 'INFO',
    })
    
    runner = CrawlerRunner(settings)

    @defer.inlineCallbacks
    def crawl():
        yield runner.crawl(NABARDSpider)
        yield runner.crawl(SBISpider)
        
        general_targets = [
            
            {'issuing_body': 'National Portal of India', 'start_urls': 'https://www.india.gov.in/topics/agriculture', 'allowed_domains': 'india.gov.in', 'allow_rules': ('agriculture', 'scheme', 'loan', 'subsidy')},
          
        ]

        for target in general_targets:
            yield runner.crawl(GeneralSpider, **target)
        
        reactor.stop()

    # --- Setup and run animation thread ---
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=animate, args=(stop_spinner,))
    spinner_thread.start()

    crawl()
    reactor.run() # The script will block here until the crawls are finished

    # --- Stop animation and proceed ---
    stop_spinner.set()
    spinner_thread.join()
    
    if Path(output_file).exists() and Path(output_file).stat().st_size > 0:
        logging.info(f"--- Starting Data Processing from {output_file} ---")
        processor = DataProcessor()
        processor.process_scraped_data(output_file)
        logging.info("--- Data Processing Completed! ---")
    else:
        logging.warning("No data was scraped. Skipping processing.")

if __name__ == '__main__':
    run_scrapers()

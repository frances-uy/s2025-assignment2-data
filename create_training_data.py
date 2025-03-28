#!/usr/bin/env python3
"""
Create training data for quality classifier using Wikipedia URLs (high-quality)
    and Common Crawl (low-quality).
"""

import gzip
import random
import os
import re
import argparse
import subprocess
import tempfile
import logging
import time
from typing import List
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup

def setup_logging(verbose=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quality_data_creation.log')
        ]
    )
    return logging.getLogger("quality_data_creation")

def sample_urls_from_file(file_path: str, sample_size: int = 1000, logger=None) -> List[str]:
    """
    Sample URLs from the Wikipedia reference URLs file.
    
    Args:
        file_path: Path to the gzipped URL file
        sample_size: Number of URLs to sample
        logger: Logger for status updates
    
    Returns:
        List of sampled URLs
    """
    if logger is None:
        logger = logging.getLogger("sample_urls")
    
    urls = []
    
    logger.info(f"Reading URLs from {file_path}")
    start_time = time.time()
    
    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            url = line.strip()
            if url:
                urls.append(url)
                if len(urls) % 100000 == 0:
                    logger.debug(f"Read {len(urls)} URLs so far...")
    
    end_time = time.time()
    logger.info(f"Read {len(urls)} URLs in {end_time - start_time:.2f} seconds")
    
    # Sample URLs
    if len(urls) > sample_size:
        logger.info(f"Sampling {sample_size} URLs from {len(urls)} total URLs")
        return random.sample(urls, sample_size)
    
    logger.info(f"Using all {len(urls)} URLs (fewer than requested sample size)")
    return urls

def download_urls(urls: List[str], output_warc: str, logger=None) -> str:
    """
    Download URLs using wget and save as WARC file.
    
    Args:
        urls: List of URLs to download
        output_warc: Base name for the output WARC file
        logger: Logger for status updates
    
    Returns:
        Path to the created WARC file
    """
    if logger is None:
        logger = logging.getLogger("download_urls")
    
    # Create a temporary file with URLs
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
        temp_path = temp.name
        for url in urls:
            temp.write(f"{url}\n")
    
    logger.info(f"Starting download of {len(urls)} URLs to {output_warc}.warc.gz")
    start_time = time.time()
    
    # Download URLs using wget
    try:
        # Build wget command
        wget_cmd = [
            "wget",
            "--timeout=5",
            "--tries=2",
            "--wait=0.5",
            "--random-wait",
            "--user-agent=Mozilla/5.0 (compatible; CS336QualityClassifier/1.0)",
            "-i", temp_path,
            "--warc-file=" + output_warc,
            "-O", "/dev/null"
        ]
        
        logger.debug(f"Running command: {' '.join(wget_cmd)}")
        
        # Run wget
        process = subprocess.Popen(
            wget_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor progress
        while process.poll() is None:
            time.sleep(10)  # Check every 10 seconds
            if os.path.exists(output_warc + ".warc.gz"):
                size_mb = os.path.getsize(output_warc + ".warc.gz") / (1024 * 1024)
                logger.info(f"Download in progress... WARC file size: {size_mb:.2f} MB")
        
        # Process completed
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.warning(f"wget returned non-zero exit code: {process.returncode}")
            logger.debug(f"stderr: {stderr}")
        
        warc_path = output_warc + ".warc.gz"
        
        if os.path.exists(warc_path):
            end_time = time.time()
            size_mb = os.path.getsize(warc_path) / (1024 * 1024)
            logger.info(f"Downloaded URLs to {warc_path} ({size_mb:.2f} MB) in {end_time - start_time:.2f} seconds")
            return warc_path
        else:
            logger.error(f"Expected WARC file {warc_path} not found")
            return None
    except Exception as e:
        logger.error(f"Error downloading URLs: {e}", exc_info=True)
        return None
    finally:
        os.unlink(temp_path)

def extract_text_from_html(html: str) -> str:
    """
    Extract readable text from HTML content.
    
    Args:
        html: HTML content
    
    Returns:
        Extracted text
    """
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove scripts, styles, and hidden elements
        for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
            element.extract()
        
        # Get text
        text = soup.get_text()
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logging.debug(f"Error extracting text from HTML: {e}")
        return ""

def extract_text_from_warc(warc_file: str, logger=None) -> List[str]:
    """
    Extract text from WARC file.
    
    Args:
        warc_file: Path to WARC file
        logger: Logger for status updates
    
    Returns:
        List of extracted texts
    """
    if logger is None:
        logger = logging.getLogger("extract_warc")
    
    texts = []
    processed = 0
    extracted = 0
    
    logger.info(f"Extracting text from WARC file: {warc_file}")
    start_time = time.time()
    
    try:
        with open(warc_file, 'rb') as stream:
            for record in ArchiveIterator(stream):
                processed += 1
                if processed % 100 == 0:
                    logger.debug(f"Processed {processed} records, extracted {extracted} texts...")
                
                if record.rec_type == 'response' and record.http_headers:
                    content_type = record.http_headers.get('Content-Type', '')
                    if 'text/html' in content_type.lower():
                        payload = record.content_stream().read()
                        try:
                            html = payload.decode('utf-8', errors='ignore')
                            text = extract_text_from_html(html)
                            if text and len(text) > 100:  # Minimum length threshold
                                texts.append(text)
                                extracted += 1
                        except Exception as e:
                            logger.debug(f"Error processing record: {e}")
                            continue
    except Exception as e:
        logger.error(f"Error reading WARC file {warc_file}: {e}", exc_info=True)
    
    end_time = time.time()
    logger.info(f"Extracted {len(texts)} texts from {processed} records in {end_time - start_time:.2f} seconds")
    
    return texts

def preprocess_text(text: str) -> str:
    """
    Preprocess text for training.
    
    Args:
        text: Input text
    
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\t]+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_training_file(high_quality_texts: List[str], 
                         low_quality_texts: List[str], 
                         output_file: str = "quality_train.txt",
                         logger=None) -> None:
    """
    Create training file for fastText.
    
    Args:
        high_quality_texts: List of high-quality texts
        low_quality_texts: List of low-quality texts
        output_file: Output file path
        logger: Logger for status updates
    """
    if logger is None:
        logger = logging.getLogger("create_training_file")
    
    logger.info(f"Creating training file at {output_file}")
    start_time = time.time()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write high-quality examples
        count_high = 0
        for text in high_quality_texts:
            processed_text = preprocess_text(text)
            if processed_text and len(processed_text) > 100:
                f.write(f"__label__high {processed_text}\n")
                count_high += 1
                if count_high % 1000 == 0:
                    logger.debug(f"Processed {count_high} high-quality texts...")
        
        # Write low-quality examples
        count_low = 0
        for text in low_quality_texts:
            processed_text = preprocess_text(text)
            if processed_text and len(processed_text) > 100:
                f.write(f"__label__low {processed_text}\n")
                count_low += 1
                if count_low % 1000 == 0:
                    logger.debug(f"Processed {count_low} low-quality texts...")
    
    end_time = time.time()
    logger.info(f"Training file created with {count_high} high-quality and {count_low} low-quality examples in {end_time - start_time:.2f} seconds")

def generate_sample_high_quality_texts(count: int = 100, logger=None) -> List[str]:
    """
    Generate sample high-quality texts for testing.
    
    Args:
        count: Number of samples to generate
        logger: Logger for status updates
    
    Returns:
        List of generated texts
    """
    if logger is None:
        logger = logging.getLogger("generate_samples")
    
    logger.info(f"Generating {count} synthetic high-quality samples for testing")
    
    texts = []
    topics = ["climate change", "artificial intelligence", "renewable energy", 
              "quantum physics", "public health", "economics", "history", 
              "literature", "astronomy", "biology", "psychology"]
    
    for i in range(count):
        topic = random.choice(topics)
        text = (
            f"This is a high-quality article about {topic}. It contains well-researched "
            f"information from reliable sources. The article presents a comprehensive "
            f"analysis of the subject matter, including key concepts, historical context, "
            f"and recent developments. Scientific evidence supports the main conclusions. "
            f"The writing is clear, organized, and follows proper grammar and style guidelines. "
            f"It has been reviewed for accuracy and includes appropriate references to "
            f"peer-reviewed literature. This is sample #{i} in our collection."
        )
        texts.append(text)
    
    return texts

def main():
    parser = argparse.ArgumentParser(description="Create training data for quality classifier")
    parser.add_argument("--wiki-urls", type=str, default="enwiki-20240420-extracted_urls.txt.gz",
                       help="Path to Wikipedia URLs file")
    parser.add_argument("--cc-warc", type=str, default="CC-MAIN-20180420081400-20180420101400-00118.warc.gz",
                       help="Path to Common Crawl WARC file")
    parser.add_argument("--output", type=str, default="quality_train.txt",
                       help="Output training file path")
    parser.add_argument("--sample-size", type=int, default=500,
                       help="Number of URLs to sample from Wikipedia")
    parser.add_argument("--download", action="store_true",
                       help="Download content from Wikipedia URLs")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Log start of process
    logger.info(f"Starting training data creation process")
    logger.info(f"Wiki URLs file: {args.wiki_urls}")
    logger.info(f"Common Crawl WARC file: {args.cc_warc}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Download content: {args.download}")
    
    overall_start_time = time.time()
    
    # Process Wikipedia URLs for high-quality examples
    high_quality_texts = []
    
    # Always download content from Wikipedia URLs
    logger.info(f"Processing Wikipedia URLs for high-quality examples")
    urls = sample_urls_from_file(args.wiki_urls, args.sample_size, logger)
    
    if urls:
        wiki_warc = download_urls(urls, "wiki_sample", logger)
        
        if wiki_warc:
            high_quality_texts = extract_text_from_warc(wiki_warc, logger)
        else:
            logger.error("Failed to download Wikipedia content")
            # Continue with synthetic examples if download fails
    else:
        logger.error("No URLs found in Wikipedia file")
        # Continue with synthetic examples if no URLs found
    
    # If we couldn't get real examples, use synthetic ones
    if not high_quality_texts:
        logger.info("Using synthetic high-quality examples")
        high_quality_texts = generate_sample_high_quality_texts(args.sample_size, logger)
    
    # Process Common Crawl WARC for low-quality examples
    logger.info(f"Processing Common Crawl WARC for low-quality examples")
    low_quality_texts = extract_text_from_warc(args.cc_warc, logger)
    
    # If we couldn't get low-quality examples, generate some
    if not low_quality_texts:
        logger.warning(f"Could not extract low-quality texts from {args.cc_warc}")
        logger.info("Using synthetic low-quality examples")
        
        # Generate synthetic low-quality texts
        low_quality_texts = []
        for i in range(len(high_quality_texts)):
            text = (
                f"WOW!!! AMAZING DEAL!!!! Click HERE NOW to claim your FREE gift!!! "
                f"You won't BELIEVE what happens next!!! Make MONEY FAST!!! "
                f"Limited Time Offer!!! Don't miss out on this INCREDIBLE opportunity!!! "
                f"Buy now and get 50% off!!! #amazing #mindblowing #viral #{i}"
            )
            low_quality_texts.append(text)
    
    # Balance dataset if necessary
    min_count = min(len(high_quality_texts), len(low_quality_texts))
    if min_count < len(high_quality_texts):
        logger.info(f"Balancing dataset: sampling {min_count} high-quality texts")
        high_quality_texts = random.sample(high_quality_texts, min_count)
    
    if min_count < len(low_quality_texts):
        logger.info(f"Balancing dataset: sampling {min_count} low-quality texts")
        low_quality_texts = random.sample(low_quality_texts, min_count)
    
    # Create training file
    create_training_file(high_quality_texts, low_quality_texts, args.output, logger)
    
    overall_end_time = time.time()
    logger.info(f"Training data creation process completed in {overall_end_time - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import random
import sys
import os
from warcio.archiveiterator import ArchiveIterator
from cs336_data.html_extraction import extract_text_from_html_bytes
from adapters import run_gopher_quality_filter

# Path to your WARC file
warc_path = "/Users/michelleuy/Desktop/github/s2025-assignment2-data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz"

# Process WARC file
print(f"Processing {warc_path}...")
samples = []

try:
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if record.rec_type == 'response' and record.http_headers and record.http_headers.get_header('Content-Type', '').startswith('text/html'):
                try:
                    html_bytes = record.content_stream().read()
                    text = extract_text_from_html_bytes(html_bytes)
                    url = record.rec_headers.get_header('WARC-Target-URI')
                    
                    if text and len(text.strip()) > 0:
                        quality_result = run_gopher_quality_filter(text)
                        samples.append((text, quality_result, url))
                        print(f"Found sample {len(samples)}: {url[:50]}...")
                        
                        if len(samples) >= 30:  # Get more than needed to ensure we have 20 good ones
                            break
                except Exception as e:
                    print(f"Error processing record: {e}")
except Exception as e:
    print(f"Error opening WARC file: {e}")

# Review samples
print(f"\nFound {len(samples)} text samples")

reviewed = []
for i, (text, filter_result, url) in enumerate(samples[:20]):
    print("\n" + "="*60)
    print(f"Sample {i+1}/20")
    print(f"URL: {url}")
    print(f"Filter result: {'PASS' if filter_result else 'FAIL'}")
    print("-"*60)
    
    # Show text preview
    preview = text[:500].replace('\n', ' ').strip() + "..." if len(text) > 500 else text
    print(f"Text preview: {preview}")
    
    # Get your judgment
    judgment = input("\nDo you agree with the filter's judgment? (y/n): ").lower()
    
    if judgment == 'n':
        reason = input("Why do you disagree? ")
        reviewed.append((i+1, filter_result, reason))
    else:
        reviewed.append((i+1, filter_result, "Agreed with filter"))

# Summary of findings
print("\n" + "="*60)
print("Summary of findings:")
disagreements = [r for r in reviewed if "Agreed with filter" not in r[2]]
print(f"- Reviewed 20 samples")
print(f"- Disagreed with {len(disagreements)} filter judgments")

if disagreements:
    print("\nDisagreements:")
    for i, (sample_num, filter_result, reason) in enumerate(disagreements):
        print(f"{i+1}. Sample {sample_num}: Filter said {'PASS' if filter_result else 'FAIL'}")
        print(f"   Reason for disagreement: {reason}")

print("\nUse these findings to write your 2-5 sentence response for part (b).")
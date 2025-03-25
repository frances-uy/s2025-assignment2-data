import subprocess
import tempfile
import os
import sys

# Add paths to ensure imports work
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'cs336-data'))

# Import from your module
from cs336_data.html_extraction import extract_text_from_html_bytes

def get_html_from_warc(warc_path):
    """Extract HTML content from the first response record in a WARC file using gunzip."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        subprocess.run(["gunzip", "-c", warc_path], stdout=open(temp_path, "wb"))
        
        with open(temp_path, 'rb') as f:
            content = f.read()
        
        os.unlink(temp_path)
        
        # Look for HTTP response with HTML content
        http_responses = content.split(b'HTTP/1.')
        print(f"Found {len(http_responses)} HTTP responses")
        
        for i, response in enumerate(http_responses[1:6]):  # Skip first which is header, check next 5
            if b'Content-Type: text/html' in response[:500]:  # Check just the header part
                print(f"Found HTML response {i}")
                parts = response.split(b'\r\n\r\n', 1)
                if len(parts) > 1:
                    return parts[1]
        
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_text_from_wet(wet_path):
    """Extract text content from the first conversion record in a WET file using gunzip."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        subprocess.run(["gunzip", "-c", wet_path], stdout=open(temp_path, "wb"))
        
        with open(temp_path, 'rb') as f:
            content = f.read()
        
        os.unlink(temp_path)
        
        parts = content.split(b'WARC-Type: conversion')
        print(f"Found {len(parts)-1} conversion records")
        
        for i, part in enumerate(parts[1:3]):  # Look at first 2 conversion records
            headers_end = part.find(b'\r\n\r\n')
            if headers_end > 0:
                text_content = part[headers_end+4:]
                return text_content.decode('utf-8', errors='replace')
        
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    warc_path = "./CC-MAIN-20180420081400-20180420101400-00118.warc.gz"
    wet_path = "./CC-MAIN-20180420081400-20180420101400-00118.warc.wet.gz"
    
    print(f"Looking for WARC file at: {os.path.abspath(warc_path)}")
    print(f"Looking for WET file at: {os.path.abspath(wet_path)}")
    
    html_content = get_html_from_warc(warc_path)
    if html_content is not None:
        print("Successfully extracted HTML from WARC file")
        print(f"HTML content size: {len(html_content)} bytes")
        
        extracted_text = extract_text_from_html_bytes(html_content)
        print("Text extracted by our function (first 1000 chars):")
        print("-" * 50)
        print(extracted_text[:1000])
        print("-" * 50)
        
        wet_text = get_text_from_wet(wet_path)
        if wet_text is not None:
            print("\nText from WET file (first 1000 chars):")
            print("-" * 50)
            print(wet_text[:1000])
            print("-" * 50)
            
            print("\nComparison:")
            print(f"Our extraction length: {len(extracted_text)} characters")
            print(f"WET extraction length: {len(wet_text)} characters")
        else:
            print("Could not extract text from WET file")
    else:
        print("Could not extract HTML content from WARC file")
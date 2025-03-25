from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

def extract_text_from_html_bytes(html_bytes):
    """
    Extract plain text from HTML byte string.
    
    Args:
        html_bytes (bytes): Raw HTML content as bytes
        
    Returns:
        str: Extracted plain text
    """
    # First try UTF-8 decoding
    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, try to detect the encoding
        detected_encoding = detect_encoding(html_bytes)
        if detected_encoding:
            try:
                html_str = html_bytes.decode(detected_encoding)
            except UnicodeDecodeError:
                # If all else fails, use 'replace' to handle decoding errors
                html_str = html_bytes.decode('utf-8', errors='replace')
        else:
            # Fallback with error replacement
            html_str = html_bytes.decode('utf-8', errors='replace')
    
    # Extract plain text using Resiliparse
    extracted_text = extract_plain_text(html_str)
    return extracted_text
"""Email parsing utilities."""
import email
from email.message import Message
from email.header import decode_header
from datetime import datetime
from typing import Dict, Optional
import re

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False


def decode_mime_words(s: str) -> str:
    """Decode MIME encoded words."""
    if not s:
        return ""
    
    decoded_parts = decode_header(s)
    decoded_string = ""
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            if encoding:
                try:
                    decoded_string += part.decode(encoding)
                except:
                    decoded_string += part.decode('utf-8', errors='ignore')
            else:
                decoded_string += part.decode('utf-8', errors='ignore')
        else:
            decoded_string += str(part)
    
    return decoded_string


def get_email_body(msg: Message) -> str:
    """Extract email body, preferring text/plain over text/html."""
    body = ""
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
            
            # Prefer text/plain
            if content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        body = payload.decode(charset, errors='ignore')
                        break  # Found text/plain, use it
                except:
                    pass
        
        # If no text/plain found, try text/html
        if not body:
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if "attachment" in content_disposition:
                    continue
                
                if content_type == "text/html":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            html_body = payload.decode(charset, errors='ignore')
                            
                            # Convert HTML to text
                            if BEAUTIFULSOUP_AVAILABLE:
                                soup = BeautifulSoup(html_body, 'html.parser')
                                body = soup.get_text(separator=' ', strip=True)
                            else:
                                # Fallback: simple regex to remove HTML tags
                                body = re.sub(r'<[^>]+>', '', html_body)
                            break
                    except:
                        pass
    else:
        # Not multipart
        content_type = msg.get_content_type()
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                body = payload.decode(charset, errors='ignore')
                
                # If HTML, convert to text
                if content_type == "text/html":
                    if BEAUTIFULSOUP_AVAILABLE:
                        soup = BeautifulSoup(body, 'html.parser')
                        body = soup.get_text(separator=' ', strip=True)
                    else:
                        body = re.sub(r'<[^>]+>', '', body)
        except:
            pass
    
    # Clean up body
    body = body.strip()
    # Remove excessive whitespace
    body = re.sub(r'\s+', ' ', body)
    
    return body


def parse_email(msg: Message) -> Dict:
    """Parse email message into structured format."""
    # Subject
    subject = decode_mime_words(msg.get('Subject', ''))
    if not subject:
        subject = '(No Subject)'
    
    # Date
    date_str = msg.get('Date', '')
    date_obj = None
    if date_str:
        try:
            date_obj = email.utils.parsedate_to_datetime(date_str)
        except:
            try:
                # Fallback parsing
                date_obj = datetime.fromtimestamp(email.utils.mktime_tz(email.utils.parsedate_tz(date_str)))
            except:
                pass
    
    date_display = date_obj.strftime('%Y-%m-%d %H:%M:%S') if date_obj else date_str
    
    # Body
    body = get_email_body(msg)
    
    # Snippet (first 150 characters)
    snippet = body[:150] + '...' if len(body) > 150 else body
    if not snippet:
        snippet = '(No content)'
    
    # From
    from_addr = decode_mime_words(msg.get('From', ''))
    
    # To
    to_addr = decode_mime_words(msg.get('To', ''))
    
    return {
        'subject': subject,
        'date': date_display,
        'date_obj': date_obj,
        'body': body,
        'snippet': snippet,
        'from': from_addr,
        'to': to_addr
    }


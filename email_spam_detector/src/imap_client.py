"""IMAP client for Gmail inbox scanning."""
import imaplib
import email
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from email.header import decode_header

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Try current directory
except ImportError:
    print("Warning: python-dotenv not installed. Loading .env manually.")
    # Fallback: load .env manually
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


class IMAPClient:
    """IMAP client for connecting to Gmail and fetching emails."""
    
    def __init__(self):
        """Initialize IMAP client with credentials from environment."""
        self.email = os.getenv('GMAIL_EMAIL', '')
        self.app_password = os.getenv('GMAIL_APP_PASSWORD', '')
        self.imap_server = os.getenv('IMAP_SERVER', 'imap.gmail.com')
        self.inbox_folder = os.getenv('INBOX_FOLDER', 'INBOX')
        self.spam_folder = os.getenv('SPAM_FOLDER', '[Gmail]/Spam')
        self.fetch_limit = int(os.getenv('FETCH_LIMIT', '20'))
        
        if not self.email or self.email == 'your_email@gmail.com':
            raise ValueError("GMAIL_EMAIL must be set in .env file. Please edit .env and add your Gmail address.")
        
        if not self.app_password or self.app_password == 'your_app_password_here':
            raise ValueError("GMAIL_APP_PASSWORD must be set in .env file. Please create a Gmail App Password and add it to .env file.")
        
        self.connection = None
    
    def connect(self) -> bool:
        """Connect and login to IMAP server."""
        try:
            self.connection = imaplib.IMAP4_SSL(self.imap_server)
            self.connection.login(self.email, self.app_password)
            return True
        except imaplib.IMAP4.error as e:
            print(f"IMAP login error: {e}")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Close IMAP connection."""
        if self.connection:
            try:
                self.connection.close()
                self.connection.logout()
            except:
                pass
            self.connection = None
    
    def list_folders(self) -> List[str]:
        """List all available folders."""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            status, folders = self.connection.list()
            if status == 'OK':
                folder_names = []
                for folder in folders:
                    # Parse folder name from bytes
                    folder_str = folder.decode('utf-8')
                    # Extract folder name (format: '(\HasNoChildren) "/" "INBOX"')
                    parts = folder_str.split('"')
                    if len(parts) >= 3:
                        folder_name = parts[-2]
                        folder_names.append(folder_name)
                return folder_names
            return []
        except Exception as e:
            print(f"Error listing folders: {e}")
            return []
    
    def find_spam_folder(self) -> Optional[str]:
        """Auto-detect spam folder if SPAM_FOLDER doesn't work."""
        folders = self.list_folders()
        for folder in folders:
            if 'spam' in folder.lower():
                return folder
        return None
    
    def select_folder(self, folder_name: str) -> bool:
        """Select a folder. Auto-detect spam folder if needed."""
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            status, _ = self.connection.select(folder_name)
            if status == 'OK':
                return True
            
            # If folder not found and it's spam folder, try auto-detect
            if folder_name == self.spam_folder:
                spam_folder = self.find_spam_folder()
                if spam_folder:
                    status, _ = self.connection.select(spam_folder)
                    return status == 'OK'
            
            return False
        except Exception as e:
            print(f"Error selecting folder {folder_name}: {e}")
            return False
    
    def fetch_emails(self, folder_name: str, limit: Optional[int] = None) -> List[Dict]:
        """Fetch recent emails from a folder."""
        if not self.connection:
            if not self.connect():
                return []
        
        if not self.select_folder(folder_name):
            return []
        
        limit = limit or self.fetch_limit
        
        try:
            # Search for all emails
            status, messages = self.connection.search(None, 'ALL')
            if status != 'OK':
                return []
            
            # Get email IDs
            email_ids = messages[0].split()
            if not email_ids:
                return []
            
            # Get last N emails
            email_ids = email_ids[-limit:] if len(email_ids) > limit else email_ids
            email_ids.reverse()  # Most recent first
            
            emails = []
            for email_id in email_ids:
                try:
                    # Fetch email
                    status, msg_data = self.connection.fetch(email_id, '(RFC822)')
                    if status != 'OK' or not msg_data:
                        continue
                    
                    # Parse email
                    email_body = msg_data[0][1]
                    email_message = email.message_from_bytes(email_body)
                    
                    emails.append({
                        'id': email_id.decode('utf-8'),
                        'raw': email_message
                    })
                except Exception as e:
                    print(f"Error fetching email {email_id}: {e}")
                    continue
            
            return emails
        
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []
    
    def test_connection(self) -> Dict:
        """Test IMAP connection and return status."""
        try:
            if not self.connect():
                return {
                    'success': False,
                    'message': 'Failed to connect to IMAP server',
                    'folders': []
                }
            
            folders = self.list_folders()
            self.disconnect()
            
            return {
                'success': True,
                'message': 'Connection successful',
                'folders': folders
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'folders': []
            }


def main():
    """CLI interface for debugging."""
    import argparse
    
    parser = argparse.ArgumentParser(description='IMAP Client CLI')
    parser.add_argument('--list-folders', action='store_true', help='List all folders')
    parser.add_argument('--fetch-inbox', type=int, metavar='N', help='Fetch N emails from INBOX')
    parser.add_argument('--fetch-spam', type=int, metavar='N', help='Fetch N emails from SPAM')
    
    args = parser.parse_args()
    
    client = IMAPClient()
    
    try:
        if args.list_folders:
            if client.connect():
                folders = client.list_folders()
                print(f"\nFound {len(folders)} folders:")
                for folder in folders:
                    print(f"  - {folder}")
                client.disconnect()
            else:
                print("Failed to connect to IMAP server")
        
        elif args.fetch_inbox:
            if client.connect():
                emails = client.fetch_emails(client.inbox_folder, args.fetch_inbox)
                print(f"\nFetched {len(emails)} emails from INBOX")
                for i, email_data in enumerate(emails, 1):
                    print(f"\nEmail {i}:")
                    print(f"  ID: {email_data['id']}")
                    msg = email_data['raw']
                    subject = decode_header(msg['Subject'])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode('utf-8', errors='ignore')
                    print(f"  Subject: {subject}")
                client.disconnect()
            else:
                print("Failed to connect to IMAP server")
        
        elif args.fetch_spam:
            if client.connect():
                emails = client.fetch_emails(client.spam_folder, args.fetch_spam)
                print(f"\nFetched {len(emails)} emails from SPAM")
                for i, email_data in enumerate(emails, 1):
                    print(f"\nEmail {i}:")
                    print(f"  ID: {email_data['id']}")
                    msg = email_data['raw']
                    subject = decode_header(msg['Subject'])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode('utf-8', errors='ignore')
                    print(f"  Subject: {subject}")
                client.disconnect()
            else:
                print("Failed to connect to IMAP server")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


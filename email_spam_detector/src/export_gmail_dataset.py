"""Export Gmail emails to dataset CSV format."""
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from imap_client import IMAPClient
from email_parser import parse_email
from utils import clean_text, get_project_root


def export_gmail_dataset(inbox_limit=1000, spam_limit=1000, output_filename=None):
    """
    Export Gmail emails to CSV dataset format.
    
    Args:
        inbox_limit: Number of emails to fetch from INBOX (default: 1000)
        spam_limit: Number of emails to fetch from SPAM (default: 1000)
        output_filename: Output CSV filename (default: gmail_dataset_YYYYMMDD.csv)
    
    Returns:
        Path to saved CSV file
    """
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"gmail_dataset_{timestamp}.csv"
    
    output_file = raw_data_dir / output_filename
    
    print("=" * 80)
    print("GMAIL DATASET EXPORT")
    print("=" * 80)
    print()
    print(f"Target: {inbox_limit} INBOX emails + {spam_limit} SPAM emails = {inbox_limit + spam_limit} total")
    print()
    
    # Connect to Gmail
    try:
        client = IMAPClient()
        if not client.connect():
            print("❌ Failed to connect to Gmail")
            return None
        
        print("✓ Connected to Gmail")
        print()
    except Exception as e:
        print(f"❌ Error connecting to Gmail: {e}")
        return None
    
    all_emails = []
    
    # Fetch INBOX emails (label = 0 = ham)
    print(f"Fetching {inbox_limit} emails from INBOX...")
    try:
        inbox_emails = client.fetch_emails(client.inbox_folder, inbox_limit)
        print(f"  ✓ Fetched {len(inbox_emails)} emails from INBOX")
        
        for i, email_data in enumerate(inbox_emails, 1):
            try:
                parsed = parse_email(email_data['raw'])
                text = f"{parsed['subject']} {parsed['body']}"
                all_emails.append({
                    'text': text,
                    'label': 0,  # INBOX = ham
                    'source_file': output_filename
                })
                if i % 100 == 0:
                    print(f"    Processed {i}/{len(inbox_emails)} INBOX emails...")
            except Exception as e:
                print(f"  ⚠️ Error parsing INBOX email {i}: {e}")
                continue
    except Exception as e:
        print(f"  ❌ Error fetching INBOX emails: {e}")
    
    # Fetch SPAM emails (label = 1 = spam)
    print(f"\nFetching {spam_limit} emails from SPAM...")
    try:
        spam_emails = client.fetch_emails(client.spam_folder, spam_limit)
        print(f"  ✓ Fetched {len(spam_emails)} emails from SPAM")
        
        for i, email_data in enumerate(spam_emails, 1):
            try:
                parsed = parse_email(email_data['raw'])
                text = f"{parsed['subject']} {parsed['body']}"
                all_emails.append({
                    'text': text,
                    'label': 1,  # SPAM = spam
                    'source_file': output_filename
                })
                if i % 100 == 0:
                    print(f"    Processed {i}/{len(spam_emails)} SPAM emails...")
            except Exception as e:
                print(f"  ⚠️ Error parsing SPAM email {i}: {e}")
                continue
    except Exception as e:
        print(f"  ❌ Error fetching SPAM emails: {e}")
    
    client.disconnect()
    
    if not all_emails:
        print("\n❌ No emails fetched")
        return None
    
    print(f"\n✓ Total emails fetched: {len(all_emails)}")
    print()
    
    # Create DataFrame
    df = pd.DataFrame(all_emails)
    
    # Clean text
    print("Cleaning text...")
    df['text_cleaned'] = df['text'].apply(clean_text)
    
    # Remove empty rows
    before = len(df)
    df = df[df['text_cleaned'].str.len() > 0]
    after = len(df)
    print(f"  ✓ Removed {before - after} empty rows")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['text_cleaned'], keep='first')
    after = len(df)
    print(f"  ✓ Removed {before - after} duplicate rows")
    
    # Final format
    final_df = df[['text_cleaned', 'label', 'source_file']].copy()
    final_df.columns = ['text', 'label', 'source_file']
    final_df['label'] = final_df['label'].astype(int)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    
    print()
    print("=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(f"Total emails: {len(final_df)}")
    print(f"  - Ham (0): {len(final_df[final_df['label'] == 0])}")
    print(f"  - Spam (1): {len(final_df[final_df['label'] == 1])}")
    print(f"\nDataset saved to: {output_file}")
    print("=" * 80)
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export Gmail emails to dataset CSV')
    parser.add_argument('--inbox', type=int, default=1000, help='Number of emails from INBOX (default: 1000)')
    parser.add_argument('--spam', type=int, default=1000, help='Number of emails from SPAM (default: 1000)')
    parser.add_argument('--output', type=str, help='Output filename (default: gmail_dataset_YYYYMMDD.csv)')
    
    args = parser.parse_args()
    
    export_gmail_dataset(
        inbox_limit=args.inbox,
        spam_limit=args.spam,
        output_filename=args.output
    )


# Dataset Description

## Custom Dataset Collection (Gmail)

**This project uses ONLY custom dataset collected from personal Gmail account.**

### Collection Method
- **Source**: Personal Gmail account via IMAP protocol
- **Collection Date**: January 8, 2025
- **Collection Process**:
  1. Connected to Gmail using IMAP (Internet Message Access Protocol)
  2. Fetched emails from two folders:
     - **INBOX folder**: Emails labeled as **ham (0)** - legitimate emails
     - **SPAM folder**: Emails labeled as **spam (1)** - spam emails
  3. Parsed email content (subject + body) using email parsing library
  4. Applied text cleaning (lowercase, URL removal, etc.)
  5. Removed duplicates and empty entries
  6. Saved in CSV format

### Dataset Statistics
- **Total samples**: 1045
- **Ham (0)**: 952 samples (91.1%)
- **Spam (1)**: 93 samples (8.9%)

### Source File

- `gmail_dataset_20260108_185141.csv`

### Data Collection Code

The Gmail dataset was collected using `src/export_gmail_dataset.py`:
```bash
python -m src.export_gmail_dataset --inbox 1000 --spam 1000
```

This script:
- Connects to Gmail via IMAP
- Fetches emails from INBOX and SPAM folders
- Parses email content (subject + body)
- Applies text cleaning
- Saves to CSV in `data/raw/` directory

## Label Mapping

All labels were normalized to binary format:
- `0` = ham / not spam
- `1` = spam

## Cleaning Steps Applied

The following text cleaning steps were applied to all email/message content:

1. **Lowercase conversion**: All text converted to lowercase
2. **URL removal**: All URLs (http://, https://, www.) removed
3. **Email removal**: Email addresses removed
4. **Phone number removal**: Phone numbers in various formats removed
5. **Whitespace collapse**: Multiple whitespace characters collapsed to single space
6. **Empty row removal**: Rows with empty text after cleaning removed
7. **Duplicate removal**: Exact duplicate texts (based on cleaned text) removed

## Dataset Features

- **Text Column**: Combined subject + body content
- **Label Column**: Binary (0 = ham, 1 = spam)
- **Source File**: gmail_dataset_YYYYMMDD_HHMMSS.csv

## Important Note

**All data was collected by our own from personal Gmail account.**
No public datasets were used in this project.

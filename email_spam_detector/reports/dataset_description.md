# Dataset Description

## Source Files

The following CSV files were used to create the unified dataset:

- `_w1998.csv`
- `abdallah.csv`
- `kucev.csv`

## Label Mapping

All labels were normalized to binary format:
- `0` = ham / not spam
- `1` = spam

### Label Distribution

- **Ham (0)**: 8736 samples
- **Spam (1)**: 2016 samples
- **Total**: 10752 samples

## Cleaning Steps Applied

The following text cleaning steps were applied to all email/message content:

1. **Lowercase conversion**: All text converted to lowercase
2. **URL removal**: All URLs (http://, https://, www.) removed
3. **Email removal**: Email addresses removed
4. **Phone number removal**: Phone numbers in various formats removed
5. **Whitespace collapse**: Multiple whitespace characters collapsed to single space
6. **Empty row removal**: Rows with empty text after cleaning removed
7. **Duplicate removal**: Exact duplicate texts (based on cleaned text) removed

## Dataset Statistics

- **Total samples**: 10752
- **Features**: text, label, source_file
- **Class balance**: 81.2% ham, 18.8% spam

## Merging Rationale

Multiple datasets were merged to create a larger, more diverse training set. 
Each row retains a `source_file` column indicating its origin for traceability. 
The merging process ensures consistent column names and label encoding across all sources.

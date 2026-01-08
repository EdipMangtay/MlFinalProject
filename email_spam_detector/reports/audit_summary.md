# Dataset Audit Summary

## Overview
This audit was performed on 3 CSV file(s) found in `data/raw/`.

## Files Analyzed

### _w1998.csv
- **Shape**: 5728x2
- **Columns**: text, spam
- **Text Column**: text
- **Label Column**: spam
- **Normalized Mapping**: 0 = ham, 1 = spam

### abdallah.csv
- **Shape**: 5572x2
- **Columns**: Category, Message
- **Text Column**: Message
- **Label Column**: Category
- **Normalized Mapping**: ham/not spam -> 0, spam -> 1

### kucev.csv
- **Shape**: 84x3
- **Columns**: title, text, type
- **Text Column**: title
- **Label Column**: type
- **Normalized Mapping**: ham/not spam -> 0, spam -> 1

## Merging Strategy

All detected CSV files will be merged into a single unified dataset with the following columns:
- `text`: The email/message content
- `label`: Binary label (0 = ham, 1 = spam)
- `source_file`: Original CSV file name

## Label Normalization

All labels will be normalized to:
- `0` = ham / not spam
- `1` = spam

## Text Cleaning

The following cleaning steps will be applied:
1. Convert to lowercase
2. Remove URLs
3. Remove email addresses
4. Remove phone numbers
5. Collapse whitespace
6. Drop empty rows
7. Remove exact duplicates (based on cleaned text)

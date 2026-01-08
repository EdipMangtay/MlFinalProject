"""Phase 0: Dataset audit script."""
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_project_root


def audit_datasets():
    """Audit all CSV files in the workspace and generate audit summary."""
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"
    reports_dir = project_root / "reports"
    
    # Ensure reports directory exists
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(raw_data_dir.glob("*.csv"))
    
    if not csv_files:
        print("ERROR: No CSV files found in data/raw/")
        return
    
    audit_results = []
    
    print("=" * 80)
    print("PHASE 0 - DATASET AUDIT")
    print("=" * 80)
    print()
    
    for csv_file in csv_files:
        print(f"Analyzing: {csv_file.name}")
        print("-" * 80)
        
        try:
            df = pd.read_csv(csv_file)
            shape = df.shape
            columns = list(df.columns)
            
            print(f"  Shape: {shape[0]} rows, {shape[1]} columns")
            print(f"  Columns: {columns}")
            
            # Detect text column candidate
            text_candidates = ['text', 'message', 'email', 'content', 'body', 'title']
            text_col = None
            for col in columns:
                if col.lower() in text_candidates:
                    text_col = col
                    break
            if not text_col and len(columns) >= 2:
                # Assume second column if no match
                text_col = columns[1] if columns[0].lower() in ['category', 'type', 'label', 'spam', 'class'] else columns[0]
            
            # Detect label column candidate
            label_candidates = ['spam', 'label', 'category', 'type', 'class']
            label_col = None
            for col in columns:
                if col.lower() in label_candidates:
                    label_col = col
                    break
            if not label_col:
                # Try to find numeric or categorical column
                for col in columns:
                    if col != text_col:
                        unique_vals = df[col].unique()
                        if len(unique_vals) <= 10:  # Likely a label column
                            label_col = col
                            break
            
            print(f"  Text column candidate: {text_col}")
            print(f"  Label column candidate: {label_col}")
            
            if label_col:
                label_values = df[label_col].value_counts()
                print(f"  Label distribution:")
                for val, count in label_values.items():
                    print(f"    {val}: {count}")
                
                # Detect label mapping
                unique_labels = df[label_col].unique()
                label_mapping = {}
                if set(unique_labels).issubset({0, 1, '0', '1'}):
                    # Already numeric
                    label_mapping = {0: 0, 1: 1, '0': 0, '1': 1}
                    normalized = "0 = ham, 1 = spam"
                elif 'spam' in str(unique_labels).lower() and ('ham' in str(unique_labels).lower() or 'not spam' in str(unique_labels).lower()):
                    # String labels
                    if 'ham' in str(unique_labels).lower():
                        label_mapping = {'ham': 0, 'spam': 1, 'Ham': 0, 'Spam': 1, 'HAM': 0, 'SPAM': 1}
                    elif 'not spam' in str(unique_labels).lower():
                        label_mapping = {'not spam': 0, 'spam': 1, 'Not spam': 0, 'Spam': 1}
                    normalized = "ham/not spam -> 0, spam -> 1"
                else:
                    normalized = "TO BE DETERMINED"
                
                print(f"  Normalized mapping: {normalized}")
            else:
                print("  WARNING: Could not detect label column")
                normalized = "TO BE DETERMINED"
            
            audit_results.append({
                'file': csv_file.name,
                'shape': f"{shape[0]}x{shape[1]}",
                'columns': ', '.join(columns),
                'text_column': text_col,
                'label_column': label_col,
                'normalized_mapping': normalized
            })
            
        except Exception as e:
            print(f"  ERROR reading file: {e}")
            audit_results.append({
                'file': csv_file.name,
                'shape': 'ERROR',
                'columns': 'ERROR',
                'text_column': 'ERROR',
                'label_column': 'ERROR',
                'normalized_mapping': 'ERROR'
            })
        
        print()
    
    # Generate audit summary markdown
    summary_lines = [
        "# Dataset Audit Summary",
        "",
        "## Overview",
        f"This audit was performed on {len(csv_files)} CSV file(s) found in `data/raw/`.",
        "",
        "## Files Analyzed",
        ""
    ]
    
    for result in audit_results:
        summary_lines.extend([
            f"### {result['file']}",
            f"- **Shape**: {result['shape']}",
            f"- **Columns**: {result['columns']}",
            f"- **Text Column**: {result['text_column']}",
            f"- **Label Column**: {result['label_column']}",
            f"- **Normalized Mapping**: {result['normalized_mapping']}",
            ""
        ])
    
    summary_lines.extend([
        "## Merging Strategy",
        "",
        "All detected CSV files will be merged into a single unified dataset with the following columns:",
        "- `text`: The email/message content",
        "- `label`: Binary label (0 = ham, 1 = spam)",
        "- `source_file`: Original CSV file name",
        "",
        "## Label Normalization",
        "",
        "All labels will be normalized to:",
        "- `0` = ham / not spam",
        "- `1` = spam",
        "",
        "## Text Cleaning",
        "",
        "The following cleaning steps will be applied:",
        "1. Convert to lowercase",
        "2. Remove URLs",
        "3. Remove email addresses",
        "4. Remove phone numbers",
        "5. Collapse whitespace",
        "6. Drop empty rows",
        "7. Remove exact duplicates (based on cleaned text)",
        ""
    ])
    
    summary_content = "\n".join(summary_lines)
    
    # Save to file
    audit_file = reports_dir / "audit_summary.md"
    with open(audit_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print("=" * 80)
    print(f"Audit summary saved to: {audit_file}")
    print("=" * 80)
    
    return audit_results


if __name__ == "__main__":
    audit_datasets()


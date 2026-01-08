"""Dataset preparation: load, merge, clean, and save unified dataset."""
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_project_root, clean_text


def prepare_dataset():
    """Load all CSVs, merge, clean, and save unified dataset."""
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"
    data_dir = project_root / "data"
    reports_dir = project_root / "reports"
    
    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(raw_data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/raw/")
    
    print("=" * 80)
    print("DATASET PREPARATION")
    print("=" * 80)
    print()
    
    all_dataframes = []
    
    for csv_file in csv_files:
        print(f"Loading: {csv_file.name}")
        df = pd.read_csv(csv_file)
        original_shape = df.shape
        
        # Detect and extract text and label columns
        text_col = None
        label_col = None
        
        # Detect text column
        text_candidates = ['text', 'message', 'email', 'content', 'body']
        for col in df.columns:
            if col.lower() in text_candidates:
                text_col = col
                break
        
        # For kucev.csv, combine title and text
        if csv_file.name == 'kucev.csv' and 'title' in df.columns and 'text' in df.columns:
            df['combined_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
            text_col = 'combined_text'
        elif not text_col:
            # Fallback: assume second column or first non-label column
            label_candidates = ['spam', 'label', 'category', 'type', 'class']
            for col in df.columns:
                if col.lower() in label_candidates:
                    label_col = col
                    break
            if label_col:
                text_col = [c for c in df.columns if c != label_col][0] if len(df.columns) > 1 else df.columns[0]
            else:
                text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Detect label column
        label_candidates = ['spam', 'label', 'category', 'type', 'class']
        for col in df.columns:
            if col.lower() in label_candidates:
                label_col = col
                break
        
        if not label_col:
            # Try to find it by exclusion
            other_cols = [c for c in df.columns if c != text_col]
            if other_cols:
                label_col = other_cols[0]
        
        print(f"  Text column: {text_col}")
        print(f"  Label column: {label_col}")
        
        # Extract and normalize
        subset_df = pd.DataFrame()
        subset_df['text'] = df[text_col].astype(str)
        subset_df['label'] = df[label_col]
        subset_df['source_file'] = csv_file.name
        
        # Normalize labels
        unique_labels = subset_df['label'].unique()
        print(f"  Original labels: {set(unique_labels)}")
        
        # Map labels to 0 (ham) and 1 (spam)
        label_mapping = {}
        if set(str(v).lower() for v in unique_labels).issubset({'0', '1', '0.0', '1.0'}):
            # Already numeric
            subset_df['label'] = subset_df['label'].astype(str).str.strip().replace({'0': 0, '1': 1, '0.0': 0, '1.0': 1})
        elif any('spam' in str(v).lower() for v in unique_labels):
            # String labels
            if any('ham' in str(v).lower() for v in unique_labels):
                subset_df['label'] = subset_df['label'].astype(str).str.lower().replace({'ham': 0, 'spam': 1})
            elif any('not spam' in str(v).lower() for v in unique_labels):
                subset_df['label'] = subset_df['label'].astype(str).str.lower().replace({'not spam': 0, 'spam': 1})
            else:
                # Assume binary: first unique value = 0, second = 1
                unique_vals = subset_df['label'].unique()
                if len(unique_vals) == 2:
                    mapping = {str(unique_vals[0]): 0, str(unique_vals[1]): 1}
                    subset_df['label'] = subset_df['label'].astype(str).replace(mapping)
        
        subset_df['label'] = pd.to_numeric(subset_df['label'], errors='coerce')
        
        print(f"  Normalized labels: {subset_df['label'].value_counts().to_dict()}")
        print(f"  Rows: {len(subset_df)}")
        print()
        
        all_dataframes.append(subset_df)
    
    # Merge all dataframes
    print("Merging datasets...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total rows after merge: {len(merged_df)}")
    print()
    
    # Cleaning steps
    print("Cleaning dataset...")
    print("  Step 1: Applying text cleaning...")
    merged_df['text_cleaned'] = merged_df['text'].apply(clean_text)
    
    print("  Step 2: Dropping empty rows...")
    before_empty = len(merged_df)
    merged_df = merged_df[merged_df['text_cleaned'].str.len() > 0]
    after_empty = len(merged_df)
    print(f"    Dropped {before_empty - after_empty} empty rows")
    
    print("  Step 3: Removing exact duplicates...")
    before_dup = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['text_cleaned'], keep='first')
    after_dup = len(merged_df)
    print(f"    Dropped {before_dup - after_dup} duplicate rows")
    
    print("  Step 4: Dropping rows with missing labels...")
    before_label = len(merged_df)
    merged_df = merged_df.dropna(subset=['label'])
    after_label = len(merged_df)
    print(f"    Dropped {before_label - after_label} rows with missing labels")
    
    # Final dataset
    final_df = merged_df[['text_cleaned', 'label', 'source_file']].copy()
    final_df.columns = ['text', 'label', 'source_file']
    final_df['label'] = final_df['label'].astype(int)
    
    # Save to CSV
    output_file = data_dir / "dataset.csv"
    final_df.to_csv(output_file, index=False)
    
    print()
    print("=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Label distribution:")
    print(final_df['label'].value_counts().sort_index())
    print(f"\nDataset saved to: {output_file}")
    print("=" * 80)
    
    # Generate dataset description
    generate_dataset_description(final_df, csv_files, reports_dir)
    
    return final_df


def generate_dataset_description(df, source_files, reports_dir):
    """Generate dataset_description.md report."""
    lines = [
        "# Dataset Description",
        "",
        "## Source Files",
        "",
        "The following CSV files were used to create the unified dataset:",
        ""
    ]
    
    for f in source_files:
        lines.append(f"- `{f.name}`")
    
    lines.extend([
        "",
        "## Label Mapping",
        "",
        "All labels were normalized to binary format:",
        "- `0` = ham / not spam",
        "- `1` = spam",
        "",
        "### Label Distribution",
        "",
        f"- **Ham (0)**: {len(df[df['label'] == 0])} samples",
        f"- **Spam (1)**: {len(df[df['label'] == 1])} samples",
        f"- **Total**: {len(df)} samples",
        "",
        "## Cleaning Steps Applied",
        "",
        "The following text cleaning steps were applied to all email/message content:",
        "",
        "1. **Lowercase conversion**: All text converted to lowercase",
        "2. **URL removal**: All URLs (http://, https://, www.) removed",
        "3. **Email removal**: Email addresses removed",
        "4. **Phone number removal**: Phone numbers in various formats removed",
        "5. **Whitespace collapse**: Multiple whitespace characters collapsed to single space",
        "6. **Empty row removal**: Rows with empty text after cleaning removed",
        "7. **Duplicate removal**: Exact duplicate texts (based on cleaned text) removed",
        "",
        "## Dataset Statistics",
        "",
        f"- **Total samples**: {len(df)}",
        f"- **Features**: text, label, source_file",
        f"- **Class balance**: {len(df[df['label'] == 0]) / len(df) * 100:.1f}% ham, {len(df[df['label'] == 1]) / len(df) * 100:.1f}% spam",
        "",
        "## Merging Rationale",
        "",
        "Multiple datasets were merged to create a larger, more diverse training set. ",
        "Each row retains a `source_file` column indicating its origin for traceability. ",
        "The merging process ensures consistent column names and label encoding across all sources.",
        ""
    ])
    
    content = "\n".join(lines)
    desc_file = reports_dir / "dataset_description.md"
    with open(desc_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Dataset description saved to: {desc_file}")


if __name__ == "__main__":
    prepare_dataset()


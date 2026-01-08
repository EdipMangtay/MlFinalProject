# Email Spam Detector

A complete, local, report-aligned Stage 3 project for spam email classification using classical ML baselines (TF-IDF + Logistic Regression / Multinomial Naive Bayes / SVM) with Stratified 5-Fold Cross-Validation.

## Features

- **Classical ML Models**: TF-IDF vectorization with Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest, Gradient Boosting, XGBoost, and Ensemble (Voting Classifier)
- **Stratified 5-Fold Cross-Validation**: Proper CV implementation with accuracy scoring
- **Comprehensive Evaluation**: CV results, test set metrics (accuracy, precision, recall, F1, AUC)
- **Explainability**: SHAP/coefficient-based explanations for model predictions
- **Modern Web UI**: Flask-based interface with automatic pipeline for Gmail email classification
- **Gmail IMAP Integration**: Scan and classify emails directly from your Gmail inbox
- **Custom Dataset Collection**: Export Gmail emails to dataset format for model training
- **Professional Reports**: Auto-generated markdown reports for dataset description, CV results, test results, and model comparison

## Project Structure

```
email_spam_detector/
├── app.py                 # Flask web application
├── requirements.txt        # Python dependencies
├── README.md             # This file
├── data/
│   ├── raw/              # Original CSV files
│   └── dataset.csv       # Final merged & cleaned dataset
├── artifacts/
│   ├── tfidf_vectorizer.pkl
│   ├── lr_model.pkl
│   ├── nb_model.pkl
│   ├── svm_model.pkl
│   └── best_model.txt
├── reports/
│   ├── audit_summary.md
│   ├── dataset_description.md
│   ├── cv_results.csv
│   ├── test_results.csv
│   ├── model_comparison.md
│   └── shap_examples.md
├── src/
│   ├── __init__.py
│   ├── audit_datasets.py
│   ├── prepare_dataset.py
│   ├── train_evaluate.py
│   ├── predict.py
│   ├── explain.py
│   ├── imap_client.py
│   ├── email_parser.py
│   └── utils.py
├── .env.example          # Environment variables template
└── .env                  # Your actual credentials (not in git)
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Setup

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gmail IMAP (Optional)

If you want to use the Gmail inbox scanning feature:

1. **Create a Gmail App Password**:
   - Go to your Google Account settings
   - Navigate to Security → 2-Step Verification → App passwords
   - Generate a new app password for "Mail" and "Other (Custom name)"
   - Copy the 16-character password

2. **Create `.env` file**:
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` file** with your credentials:
   ```env
   GMAIL_EMAIL=your_email@gmail.com
   GMAIL_APP_PASSWORD=your_16_character_app_password
   IMAP_SERVER=imap.gmail.com
   INBOX_FOLDER=INBOX
   SPAM_FOLDER=[Gmail]/Spam
   FETCH_LIMIT=20
   PORT=5000
   FLASK_DEBUG=False
   ```

   **Important**: Never commit the `.env` file to version control. It's already in `.gitignore`.

### 4. Prepare Dataset

Run the dataset preparation script to merge and clean all CSV files:

```bash
python -m src.prepare_dataset
```

This will:
- Load all CSV files from `data/raw/`
- Normalize labels to 0 (ham) and 1 (spam)
- Clean text (lowercase, remove URLs/emails/phones, collapse whitespace)
- Remove duplicates and empty rows
- Save unified dataset to `data/dataset.csv`
- Generate `reports/dataset_description.md`

### 5. Train Models

Train all models with cross-validation:

```bash
python -m src.train_evaluate
```

This will:
- Split data into 80% train / 20% test (stratified)
- Create TF-IDF vectorizer
- Train 3 models: Logistic Regression, Multinomial Naive Bayes, Linear SVM
- Perform Stratified 5-Fold CV on train set
- Evaluate on test set
- Save models to `artifacts/`
- Generate `reports/cv_results.csv`, `reports/test_results.csv`, and `reports/model_comparison.md`

### 6. Generate Explanations (Optional)

Generate SHAP examples report:

```bash
python -m src.explain
```

This creates `reports/shap_examples.md` with example explanations.

## Running the Web Application

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to: **http://localhost:5000** (or the port specified in `.env`)

### Web UI Features

1. **Single Email Check**
   - Paste email text
   - Select model (Best/LR/NB/SVM)
   - Get prediction (SPAM/NOT SPAM) with confidence
   - View top contributing tokens

2. **Batch CSV Check**
   - Upload CSV file with email texts
   - Specify text column name
   - Process all rows and view preview
   - Download predictions as CSV

3. **Gmail Inbox Scanning** (Requires `.env` configuration)
   - **Test IMAP Connection**: Verify your Gmail credentials
   - **Scan INBOX**: Fetch and classify recent emails from your inbox
   - **Scan SPAM Folder**: Fetch and classify emails from your spam folder
   - View results as cards with:
     - Subject, date, sender
     - Predicted label (SPAM/NOT SPAM) with color coding
     - Confidence probability
     - Top contributing tokens explanation
   - Cards are color-coded: red for SPAM, green for NOT SPAM

4. **Evaluation Summary**
   - View cross-validation results
   - View test set results
   - Download results as CSV

## Usage Examples

### Single Prediction (Python)

```python
from src.predict import predict_text
from src.explain import explain_text

# Predict
result = predict_text("Your email text here", model_name='best')
print(f"Label: {result['label']} (1=spam, 0=ham)")
print(f"Probability: {result['probability']:.4f}")

# Explain
explanation = explain_text("Your email text here", top_k=5)
for token_info in explanation['tokens']:
    print(f"{token_info['token']}: {token_info['impact']:.4f}")
```

### Batch Prediction (Python)

```python
from src.predict import predict_batch

texts = ["Email 1", "Email 2", "Email 3"]
results = predict_batch(texts, model_name='best')
for i, result in enumerate(results):
    print(f"Text {i+1}: {'SPAM' if result['label'] == 1 else 'HAM'}")
```

## Reports

All reports are saved in the `reports/` directory:

- **audit_summary.md**: Initial dataset audit
- **dataset_description.md**: Dataset statistics and cleaning steps
- **cv_results.csv**: Cross-validation results (mean accuracy, std)
- **test_results.csv**: Test set metrics (accuracy, precision, recall, F1, AUC)
- **model_comparison.md**: Model comparison and analysis
- **shap_examples.md**: Example explanations

## Model Selection

The best model is selected based on:
1. **Primary**: Highest cross-validation mean accuracy
2. **Tie-breaker**: Highest F1-score on test set

The best model name is saved to `artifacts/best_model.txt`.

## Technical Details

### Text Preprocessing
- Lowercase conversion
- URL removal (http://, https://, www.)
- Email address removal
- Phone number removal
- Whitespace collapse
- Duplicate removal

### TF-IDF Vectorization
- Max features: 10,000
- English stopwords removed
- N-grams: (1, 2) - unigrams and bigrams
- Min document frequency: 2
- Max document frequency: 95%

### Models
- **Logistic Regression**: L2 regularization, max_iter=1000
- **Multinomial Naive Bayes**: Default parameters
- **Linear SVM**: L2 regularization, max_iter=1000

### Evaluation
- **CV**: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- **Scoring**: Accuracy
- **Test Metrics**: Accuracy, Precision, Recall, F1, AUC (where applicable)

## Troubleshooting

### Models Not Found
If you see "Models not trained" error:
1. Ensure you've run `python -m src.train_evaluate`
2. Check that `artifacts/` directory contains `.pkl` files

### Dataset Not Found
If dataset preparation fails:
1. Ensure CSV files are in `data/raw/`
2. Check CSV format (should have text and label columns)
3. Run `python -m src.audit_datasets` to inspect files

### SHAP Not Available
SHAP is optional. If not installed, explanations will use model coefficients (for LR/SVM) or log probabilities (for NB).

### Gmail IMAP Connection Issues
If IMAP scanning fails:
1. **Verify `.env` file exists** and contains correct credentials
2. **Check Gmail App Password**: Make sure you're using an app password, not your regular Gmail password
3. **Enable IMAP in Gmail**: Go to Gmail Settings → Forwarding and POP/IMAP → Enable IMAP
4. **Test connection via CLI**:
   ```bash
   python -m src.imap_client --list-folders
   ```
5. **Check firewall/network**: Ensure port 993 (IMAP SSL) is not blocked
6. **Verify folder names**: Some Gmail accounts use different folder names. The system will auto-detect spam folders if the default doesn't work.

### CLI Debugging Tools
Test IMAP functionality from command line:
```bash
# List all folders
python -m src.imap_client --list-folders

# Fetch 5 emails from INBOX
python -m src.imap_client --fetch-inbox 5

# Fetch 5 emails from SPAM
python -m src.imap_client --fetch-spam 5
```

## License

This project is for educational purposes.

## Author

Senior ML Engineer + Python Full-Stack Developer


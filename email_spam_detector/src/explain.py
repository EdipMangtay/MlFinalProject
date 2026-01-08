"""Explainability module using SHAP or model coefficients."""
import pickle
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_project_root

# Try to import SHAP, but handle gracefully if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_model_and_vectorizer(model_name='best'):
    """Load a specific model and vectorizer."""
    project_root = get_project_root()
    artifacts_dir = project_root / "artifacts"
    
    # Load vectorizer
    vectorizer_file = artifacts_dir / "tfidf_vectorizer.pkl"
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load best model name
    best_file = artifacts_dir / "best_model.txt"
    best_model = None
    if best_file.exists():
        with open(best_file, 'r') as f:
            best_model = f.read().strip()
    
    # Determine model to use
    if model_name == 'best':
        model_name = best_model if best_model else 'lr'
    
    # Load model
    model_file = artifacts_dir / f"{model_name}_model.pkl"
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    return model, vectorizer, model_name


def explain_text(text, model_name='best', top_k=5):
    """
    Explain prediction for a text using SHAP or model coefficients.
    
    Args:
        text: Input text string
        model_name: 'lr', 'nb', 'svm', or 'best'
        top_k: Number of top tokens to return
    
    Returns:
        dict with 'tokens' (list of {token, impact, sign})
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {'tokens': [], 'error': 'Empty text provided'}
    
    try:
        model, vectorizer, actual_model_name = load_model_and_vectorizer(model_name)
        
        # Vectorize text
        text_tfidf = vectorizer.transform([text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get token impacts
        if actual_model_name == 'lr' and hasattr(model, 'coef_'):
            # Logistic Regression: use coefficients
            coef = model.coef_[0]
            feature_values = text_tfidf.toarray()[0]
            impacts = coef * feature_values
            token_impacts = list(zip(feature_names, impacts))
            token_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
        elif actual_model_name == 'svm' and hasattr(model, 'coef_'):
            # Linear SVM: use coefficients
            coef = model.coef_[0]
            feature_values = text_tfidf.toarray()[0]
            impacts = coef * feature_values
            token_impacts = list(zip(feature_names, impacts))
            token_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
        elif actual_model_name == 'nb' and hasattr(model, 'feature_log_prob_'):
            # Naive Bayes: use log probability difference
            log_prob_spam = model.feature_log_prob_[1]  # spam class
            log_prob_ham = model.feature_log_prob_[0]   # ham class
            log_odds = log_prob_spam - log_prob_ham
            feature_values = text_tfidf.toarray()[0]
            impacts = log_odds * feature_values
            token_impacts = list(zip(feature_names, impacts))
            token_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
        else:
            # Fallback: use SHAP if available
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.LinearExplainer(model, shap.sample(text_tfidf, 100))
                    shap_values = explainer.shap_values(text_tfidf)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Use spam class
                    impacts = shap_values[0]
                    token_impacts = list(zip(feature_names, impacts))
                    token_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                except:
                    # SHAP failed, use simple feature values
                    feature_values = text_tfidf.toarray()[0]
                    token_impacts = list(zip(feature_names, feature_values))
                    token_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            else:
                # Last resort: use TF-IDF values
                feature_values = text_tfidf.toarray()[0]
                token_impacts = list(zip(feature_names, feature_values))
                token_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Format top tokens
        top_tokens = []
        for token, impact in token_impacts[:top_k]:
            if abs(impact) > 1e-6:  # Only include meaningful impacts
                top_tokens.append({
                    'token': token,
                    'impact': float(impact),
                    'sign': 'positive' if impact > 0 else 'negative'
                })
        
        return {'tokens': top_tokens}
    
    except Exception as e:
        return {'tokens': [], 'error': str(e)}


def generate_shap_examples():
    """Generate SHAP examples report with sample texts from dataset."""
    project_root = get_project_root()
    data_dir = project_root / "data"
    reports_dir = project_root / "reports"
    
    # Load dataset
    dataset_file = data_dir / "dataset.csv"
    if not dataset_file.exists():
        return
    
    df = pd.read_csv(dataset_file)
    
    # Get one spam and one ham example
    spam_examples = df[df['label'] == 1]['text'].head(2).tolist()
    ham_examples = df[df['label'] == 0]['text'].head(2).tolist()
    
    lines = [
        "# SHAP Explanation Examples",
        "",
        "This report demonstrates explainability for spam email classification.",
        "",
        "## Example 1: Spam Email",
        "",
        f"**Text**:",
        "```",
        spam_examples[0][:500] + ("..." if len(spam_examples[0]) > 500 else ""),
        "```",
        "",
    ]
    
    # Explain spam example
    explanation = explain_text(spam_examples[0], top_k=10)
    if 'tokens' in explanation:
        lines.append("**Top Contributing Tokens (towards spam classification):**")
        lines.append("")
        for token_info in explanation['tokens']:
            sign_emoji = "🔴" if token_info['sign'] == 'positive' else "🟢"
            lines.append(f"- {sign_emoji} `{token_info['token']}`: {token_info['impact']:.4f}")
        lines.append("")
    
    lines.extend([
        "## Example 2: Ham Email",
        "",
        f"**Text**:",
        "```",
        ham_examples[0][:500] + ("..." if len(ham_examples[0]) > 500 else ""),
        "```",
        "",
    ])
    
    # Explain ham example
    explanation = explain_text(ham_examples[0], top_k=10)
    if 'tokens' in explanation:
        lines.append("**Top Contributing Tokens (towards ham classification):**")
        lines.append("")
        for token_info in explanation['tokens']:
            sign_emoji = "🔴" if token_info['sign'] == 'positive' else "🟢"
            lines.append(f"- {sign_emoji} `{token_info['token']}`: {token_info['impact']:.4f}")
        lines.append("")
    
    lines.extend([
        "## Explanation Method",
        "",
        "The explanation uses model coefficients (for Logistic Regression and Linear SVM) ",
        "or log probability differences (for Naive Bayes). ",
        "Positive impact values indicate tokens that push towards spam classification, ",
        "while negative values indicate tokens that push towards ham classification.",
        "",
        "## Code Example",
        "",
        "```python",
        "from src.explain import explain_text",
        "",
        "# Explain a text",
        "result = explain_text('your email text here', model_name='best', top_k=5)",
        "",
        "# Print top tokens",
        "for token_info in result['tokens']:",
        "    print(f\"{token_info['token']}: {token_info['impact']:.4f}\")",
        "```",
        ""
    ])
    
    content = "\n".join(lines)
    shap_file = reports_dir / "shap_examples.md"
    with open(shap_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"SHAP examples report saved to: {shap_file}")


if __name__ == "__main__":
    generate_shap_examples()


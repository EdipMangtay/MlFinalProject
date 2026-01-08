"""Prediction API for loaded models."""
import pickle
import numpy as np
import sys
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_project_root


def load_artifacts():
    """Load vectorizer and all models."""
    project_root = get_project_root()
    artifacts_dir = project_root / "artifacts"
    
    # Load vectorizer
    vectorizer_file = artifacts_dir / "tfidf_vectorizer.pkl"
    if not vectorizer_file.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_file}. Run train_evaluate.py first.")
    
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load models
    models = {}
    model_files = {
        'lr': 'lr_model.pkl',
        'nb': 'nb_model.pkl',
        'svm': 'svm_model.pkl',
        'rf': 'rf_model.pkl',
        'gb': 'gb_model.pkl',
        'xgb': 'xgb_model.pkl',
        'ensemble': 'ensemble_model.pkl'
    }
    
    for name, filename in model_files.items():
        model_file = artifacts_dir / filename
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load {name} model: {e}")
                continue
    
    # Load best model name
    best_file = artifacts_dir / "best_model.txt"
    best_model = None
    if best_file.exists():
        with open(best_file, 'r') as f:
            best_model = f.read().strip()
    
    return vectorizer, models, best_model


def predict_text(text, model_name='best'):
    """
    Predict label for a single text.
    
    Args:
        text: Input text string
        model_name: 'lr', 'nb', 'svm', or 'best'
    
    Returns:
        dict with 'label', 'probability', 'decision_score'
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'label': 0,
            'probability': 0.5,
            'decision_score': 0.0,
            'error': 'Empty text provided'
        }
    
    try:
        vectorizer, models, best_model = load_artifacts()
        
        # Determine which model to use
        if model_name == 'best':
            if best_model and best_model in models:
                model = models[best_model]
            else:
                # Fallback to first available model
                model = list(models.values())[0]
                model_name = list(models.keys())[0]
        elif model_name in models:
            model = models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")
        
        # Vectorize text
        text_tfidf = vectorizer.transform([text])
        
        # Predict
        label = model.predict(text_tfidf)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(text_tfidf)[0, 1]
        elif hasattr(model, 'decision_function'):
            # For SVM, use decision function and convert to probability estimate
            decision_score = model.decision_function(text_tfidf)[0]
            # Rough probability estimate: sigmoid-like transformation
            probability = 1 / (1 + np.exp(-decision_score))
        else:
            probability = 0.5  # Default if no probability available
        
        # Get decision score
        decision_score = None
        if hasattr(model, 'decision_function'):
            decision_score = float(model.decision_function(text_tfidf)[0])
        elif hasattr(model, 'predict_proba'):
            decision_score = float(probability - 0.5)  # Center around 0
        
        return {
            'label': int(label),
            'probability': float(probability),
            'decision_score': decision_score if decision_score is not None else 0.0
        }
    
    except Exception as e:
        return {
            'label': 0,
            'probability': 0.5,
            'decision_score': 0.0,
            'error': str(e)
        }


def predict_batch(texts, model_name='best'):
    """
    Predict labels for a batch of texts.
    
    Args:
        texts: List of text strings
        model_name: 'lr', 'nb', 'svm', or 'best'
    
    Returns:
        List of prediction dicts
    """
    if not texts:
        return []
    
    try:
        vectorizer, models, best_model = load_artifacts()
        
        # Determine which model to use
        if model_name == 'best':
            if best_model and best_model in models:
                model = models[best_model]
            else:
                model = list(models.values())[0]
                model_name = list(models.keys())[0]
        elif model_name in models:
            model = models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Filter empty texts
        valid_texts = [t if isinstance(t, str) and len(t.strip()) > 0 else "" for t in texts]
        
        # Vectorize
        texts_tfidf = vectorizer.transform(valid_texts)
        
        # Predict
        labels = model.predict(texts_tfidf)
        
        # Get probabilities
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(texts_tfidf)[:, 1]
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(texts_tfidf)
            probabilities = 1 / (1 + np.exp(-decision_scores))
        else:
            probabilities = np.array([0.5] * len(valid_texts))
        
        # Format results
        results = []
        for i, text in enumerate(texts):
            if not isinstance(text, str) or len(text.strip()) == 0:
                results.append({
                    'label': 0,
                    'probability': 0.5,
                    'decision_score': 0.0,
                    'error': 'Empty text'
                })
            else:
                decision_score = None
                if hasattr(model, 'decision_function'):
                    decision_score = float(decision_scores[i])
                else:
                    decision_score = float(probabilities[i] - 0.5)
                
                results.append({
                    'label': int(labels[i]),
                    'probability': float(probabilities[i]),
                    'decision_score': decision_score
                })
        
        return results
    
    except Exception as e:
        return [{'label': 0, 'probability': 0.5, 'decision_score': 0.0, 'error': str(e)}] * len(texts)


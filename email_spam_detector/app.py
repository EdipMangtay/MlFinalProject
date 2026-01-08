"""Flask web application for spam email detection."""
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import predict_text, predict_batch, load_artifacts
from explain import explain_text
from imap_client import IMAPClient
from email_parser import parse_email

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single text prediction."""
    try:
        data = request.json
        text = data.get('text', '')
        model_name = data.get('model', 'best')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Predict
        result = predict_text(text, model_name)
        
        # Get explanation
        explanation = explain_text(text, model_name, top_k=5)
        result['explanation'] = explanation.get('tokens', [])
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    """API endpoint for batch CSV prediction."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        text_column = request.form.get('text_column', 'text')
        model_name = request.form.get('model', 'best')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        if text_column not in df.columns:
            return jsonify({'error': f'Column "{text_column}" not found in CSV'}), 400
        
        # Get texts
        texts = df[text_column].astype(str).tolist()
        
        # Predict
        results = predict_batch(texts, model_name)
        
        # Add predictions to dataframe
        df['predicted_label'] = [r['label'] for r in results]
        df['predicted_probability'] = [r['probability'] for r in results]
        df['prediction'] = df['predicted_label'].map({0: 'NOT SPAM', 1: 'SPAM'})
        
        # Return preview (first 100 rows)
        preview = df.head(100).to_dict('records')
        
        # Save full results to memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return jsonify({
            'preview': preview,
            'total_rows': len(df),
            'csv_data': output.getvalue()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluation_summary', methods=['GET'])
def api_evaluation_summary():
    """API endpoint to get evaluation summary."""
    try:
        project_root = Path(__file__).parent
        reports_dir = project_root / "reports"
        
        # Load CV results
        cv_file = reports_dir / "cv_results.csv"
        cv_results = None
        if cv_file.exists():
            cv_results = pd.read_csv(cv_file).to_dict('records')
        
        # Load test results
        test_file = reports_dir / "test_results.csv"
        test_results = None
        if test_file.exists():
            test_results = pd.read_csv(test_file).to_dict('records')
        
        # Load training results
        train_file = reports_dir / "training_results.csv"
        training_results = None
        if train_file.exists():
            training_results = pd.read_csv(train_file).to_dict('records')
        
        return jsonify({
            'cv_results': cv_results,
            'test_results': test_results,
            'training_results': training_results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_cv_results', methods=['GET'])
def download_cv_results():
    """Download CV results CSV."""
    try:
        project_root = Path(__file__).parent
        reports_dir = project_root / "reports"
        cv_file = reports_dir / "cv_results.csv"
        
        if not cv_file.exists():
            return jsonify({'error': 'CV results not found'}), 404
        
        return send_file(
            cv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name='cv_results.csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_test_results', methods=['GET'])
def download_test_results():
    """Download test results CSV."""
    try:
        project_root = Path(__file__).parent
        reports_dir = project_root / "reports"
        test_file = reports_dir / "test_results.csv"
        
        if not test_file.exists():
            return jsonify({'error': 'Test results not found'}), 404
        
        return send_file(
            test_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name='test_results.csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/check_artifacts', methods=['GET'])
def check_artifacts():
    """Check if models are trained and available."""
    try:
        vectorizer, models, best_model = load_artifacts()
        return jsonify({
            'available': True,
            'models': list(models.keys()),
            'best_model': best_model
        })
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e)
        })


@app.route('/api/imap/test', methods=['POST'])
def api_imap_test():
    """Test IMAP connection."""
    try:
        client = IMAPClient()
        result = client.test_connection()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'folders': []
        }), 500


@app.route('/api/imap/scan', methods=['POST'])
def api_imap_scan():
    """Scan emails from INBOX or SPAM folder and classify them."""
    try:
        data = request.json
        folder = data.get('folder', 'INBOX')
        limit = data.get('limit', 20)
        model_name = data.get('model', 'best')
        
        # Initialize IMAP client
        client = IMAPClient()
        if not client.connect():
            return jsonify({
                'success': False,
                'message': 'Failed to connect to IMAP server',
                'emails': []
            }), 500
        
        # Determine folder name
        if folder.upper() == 'INBOX':
            folder_name = client.inbox_folder
        elif folder.upper() == 'SPAM':
            folder_name = client.spam_folder
        else:
            folder_name = folder
        
        # Fetch emails
        raw_emails = client.fetch_emails(folder_name, limit)
        client.disconnect()
        
        if not raw_emails:
            return jsonify({
                'success': True,
                'message': f'No emails found in {folder_name}',
                'emails': []
            })
        
        # Parse and classify emails
        classified_emails = []
        for email_data in raw_emails:
            try:
                # Parse email
                parsed = parse_email(email_data['raw'])
                
                # Combine subject and body for classification
                email_text = f"{parsed['subject']} {parsed['body']}"
                
                # Predict
                prediction = predict_text(email_text, model_name)
                
                # Get explanation
                explanation = explain_text(email_text, model_name, top_k=5)
                
                # Format result
                classified_emails.append({
                    'id': email_data['id'],
                    'subject': parsed['subject'],
                    'date': parsed['date'],
                    'snippet': parsed['snippet'],
                    'from': parsed['from'],
                    'body': parsed['body'],
                    'label': prediction['label'],
                    'label_text': 'SPAM' if prediction['label'] == 1 else 'NOT SPAM',
                    'probability': prediction.get('probability', None),
                    'explanation': explanation.get('tokens', [])
                })
            except Exception as e:
                print(f"Error processing email {email_data.get('id', 'unknown')}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'message': f'Scanned {len(classified_emails)} emails from {folder_name}',
            'emails': classified_emails
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'emails': []
        }), 500


@app.route('/api/imap/auto-pipeline', methods=['POST'])
def api_auto_pipeline():
    """Automatic pipeline: connect, fetch from both INBOX and SPAM, classify all."""
    try:
        data = request.json or {}
        limit = data.get('limit', 20)
        model_name = data.get('model', 'best')
        
        results = {
            'step1': {'status': 'pending', 'message': '', 'email': '', 'folders': []},
            'step2': {'status': 'pending', 'message': '', 'inbox_count': 0, 'spam_count': 0, 'total': 0},
            'step3': {'status': 'pending', 'message': '', 'processed': 0},
            'step4': {'status': 'pending', 'message': '', 'classified_emails': []}
        }
        
        # Step 1: Connect to Gmail
        try:
            client = IMAPClient()
            
            # Check if credentials are set
            if not client.email or client.email == 'your_email@gmail.com':
                results['step1'] = {
                    'status': 'error',
                    'message': 'GMAIL_EMAIL not configured in .env file',
                    'email': '',
                    'folders': []
                }
                return jsonify({'success': False, 'results': results, 'message': 'GMAIL_EMAIL not configured'}), 500
            
            if not client.app_password or client.app_password == 'your_app_password_here':
                results['step1'] = {
                    'status': 'error',
                    'message': 'GMAIL_APP_PASSWORD not configured in .env file',
                    'email': client.email,
                    'folders': []
                }
                return jsonify({'success': False, 'results': results, 'message': 'GMAIL_APP_PASSWORD not configured'}), 500
            
            if not client.connect():
                error_msg = 'Failed to connect to IMAP server. Check your Gmail App Password.'
                results['step1'] = {
                    'status': 'error',
                    'message': error_msg,
                    'email': client.email,
                    'folders': []
                }
                return jsonify({'success': False, 'results': results, 'message': error_msg}), 500
            
            folders = client.list_folders()
            results['step1'] = {
                'status': 'success',
                'message': 'Connected successfully',
                'email': client.email,
                'folders': folders
            }
        except ValueError as e:
            # This catches the ValueError from IMAPClient.__init__
            results['step1'] = {
                'status': 'error',
                'message': str(e),
                'email': '',
                'folders': []
            }
            return jsonify({'success': False, 'results': results, 'message': str(e)}), 500
        except Exception as e:
            import traceback
            error_detail = str(e)
            print(f"Connection error: {error_detail}")
            print(traceback.format_exc())
            results['step1'] = {
                'status': 'error',
                'message': f'Connection error: {error_detail}',
                'email': '',
                'folders': []
            }
            return jsonify({'success': False, 'results': results, 'message': error_detail}), 500
        
        # Step 2: Fetch emails from both INBOX and SPAM
        try:
            inbox_emails = client.fetch_emails(client.inbox_folder, limit)
            spam_emails = client.fetch_emails(client.spam_folder, limit)
            
            all_emails = inbox_emails + spam_emails
            
            results['step2'] = {
                'status': 'success',
                'message': f'Fetched {len(all_emails)} emails',
                'inbox_count': len(inbox_emails),
                'spam_count': len(spam_emails),
                'total': len(all_emails)
            }
        except Exception as e:
            results['step2'] = {
                'status': 'error',
                'message': f'Error fetching emails: {str(e)}',
                'inbox_count': 0,
                'spam_count': 0,
                'total': 0
            }
            client.disconnect()
            return jsonify({'success': False, 'results': results}), 500
        
        # Step 3: Parse emails
        try:
            parsed_emails = []
            for email_data in all_emails:
                try:
                    parsed = parse_email(email_data['raw'])
                    parsed_emails.append({
                        'id': email_data['id'],
                        'parsed': parsed,
                        'raw': email_data['raw']
                    })
                except Exception as e:
                    print(f"Error parsing email: {e}")
                    continue
            
            results['step3'] = {
                'status': 'success',
                'message': f'Processed {len(parsed_emails)} emails',
                'processed': len(parsed_emails)
            }
        except Exception as e:
            results['step3'] = {
                'status': 'error',
                'message': f'Error parsing emails: {str(e)}',
                'processed': 0
            }
            client.disconnect()
            return jsonify({'success': False, 'results': results}), 500
        
        # Step 4: Classify all emails
        try:
            classified_emails = []
            for email_item in parsed_emails:
                try:
                    parsed = email_item['parsed']
                    email_text = f"{parsed['subject']} {parsed['body']}"
                    
                    # Predict
                    prediction = predict_text(email_text, model_name)
                    
                    # Get explanation
                    explanation = explain_text(email_text, model_name, top_k=5)
                    
                    classified_emails.append({
                        'id': email_item['id'],
                        'subject': parsed['subject'],
                        'date': parsed['date'],
                        'snippet': parsed['snippet'],
                        'from': parsed['from'],
                        'body': parsed['body'],
                        'label': prediction['label'],
                        'label_text': 'SPAM' if prediction['label'] == 1 else 'NOT SPAM',
                        'probability': prediction.get('probability', None),
                        'explanation': explanation.get('tokens', [])
                    })
                except Exception as e:
                    print(f"Error classifying email: {e}")
                    continue
            
            results['step4'] = {
                'status': 'success',
                'message': f'Classified {len(classified_emails)} emails',
                'classified_emails': classified_emails
            }
        except Exception as e:
            results['step4'] = {
                'status': 'error',
                'message': f'Error classifying emails: {str(e)}',
                'classified_emails': []
            }
        
        client.disconnect()
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        import traceback
        error_detail = str(e)
        print(f"Pipeline error: {error_detail}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Pipeline error: {error_detail}',
            'results': results if 'results' in locals() else {}
        }), 500


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 80)
    print("Email Spam Detector - Web Application")
    print("=" * 80)
    print()
    print("Starting Flask server...")
    print(f"Open your browser and navigate to: http://localhost:{port}")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    app.run(debug=debug, host='0.0.0.0', port=port)


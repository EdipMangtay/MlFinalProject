"""Railway root redirect - actual app is in email_spam_detector/"""
import os
import sys
from pathlib import Path

# Change to email_spam_detector directory
email_spam_detector_path = Path(__file__).parent / "email_spam_detector"
os.chdir(email_spam_detector_path)
sys.path.insert(0, str(email_spam_detector_path))

# Import and run the actual app
from app import app

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(debug=debug, host='0.0.0.0', port=port)


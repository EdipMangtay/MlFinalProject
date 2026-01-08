#!/bin/bash
echo "Port sorununu çözüyorum..."

# Port 5000'i kullanan process'i kapat
lsof -ti:5000 2>/dev/null | xargs kill -9 2>/dev/null
echo "✓ Port 5000 temizlendi"

# Port 5001'i kullanan process'i kapat (varsa)
lsof -ti:5001 2>/dev/null | xargs kill -9 2>/dev/null

# .env dosyasını güncelle
cd /Users/mangtay/Desktop/Mlapp/email_spam_detector
sed -i '' 's/PORT=5000/PORT=5001/' .env 2>/dev/null || sed -i 's/PORT=5000/PORT=5001/' .env
echo "✓ Port 5001 olarak ayarlandı"

# Uygulamayı başlat
echo ""
echo "Uygulama başlatılıyor..."
python app.py

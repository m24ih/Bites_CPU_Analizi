# Temel imaj olarak Python 3.10 kullanıyoruz.
FROM python:3.10-slim

# Konteyner içinde çalışacağımız klasörü oluşturuyoruz.
WORKDIR /app

# Önce sadece requirements.txt dosyasını kopyalıyoruz (önbellekleme için).
COPY requirements.txt .

# Gerekli tüm kütüphaneleri kuruyoruz.
RUN pip install --no-cache-dir -r requirements.txt

# Projenin geri kalan tüm dosyalarını (app.py, data/, models/ vb.) kopyalıyoruz.
COPY . .

# Streamlit'in çalışacağı 8501 portunu dış dünyaya açıyoruz.
EXPOSE 8501

# Konteyner çalıştığında Streamlit uygulamasını başlatacak komutu ayarlıyoruz.
CMD ["streamlit", "run", "4_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# Adım 1: Temel imajı belirle
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Adım 2: Gerekli sistem paketlerini kur
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Adım 3: Çalışma dizinini ayarla
WORKDIR /app

# Adım 4: ÖNCE SADECE requirements.txt dosyasını kopyala
COPY requirements.txt .

# Adım 5: Kütüphaneleri kur
RUN pip3 install --no-cache-dir -r requirements.txt

# Adım 6: SON OLARAK projenin geri kalan tüm dosyalarını kopyala
COPY . .

# Adım 7: Varsayılan komutu ayarla
CMD ["bash"]
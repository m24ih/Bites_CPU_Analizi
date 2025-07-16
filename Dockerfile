# 1. Adım: Temel imajı beirle.
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# ENV (Ortam Değişkenleri) ayarları. Bu, kurulumların interaktif moda geçmesini engeller.
ENV DEBIAN_FRONTEND=noninteractive

# 2. Adım: Gerekli sistem paketlerini ve Python/pip'i kur.
# apt-get update ile paket listesini güncelliyoruz.
# python3 ve python3-pip'i kuruyoruz.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 3. Adım: Çalışma dizinini ayarla.
WORKDIR /app

# 4. Adım: Bağımlılık dosyasını konteynerin içine kopyala.
COPY requirements.txt .

# 5. Adım: Gerekli kütüphaneleri kur. Artık pip komutumuz çalışacak.
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Adım: Proje klasöründeki diğer tüm dosyaları konteynerin içine kopyala.
COPY . .

# 7. Adım: Konteyner başladığında çalışacak varsayılan komut.
CMD ["bash"]

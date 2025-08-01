import pandas as pd
import numpy as np

print("Anormallik tespiti için örnek veri seti oluşturuluyor...")

# --- Parametreler ---
# Oluşturulacak toplam veri noktası sayısı
num_records = 10000 
# Veri noktaları arasındaki saniye cinsinden zaman aralığı (5 dakika = 300 saniye)
interval_seconds = 300

# --- Zaman Damgalarını Oluşturma ---
# 0'dan başlayarak belirtilen aralıkla artan bir sayı dizisi oluştur
timestamps = np.arange(0, num_records * interval_seconds, interval_seconds)

# --- Normal CPU Verilerini Oluşturma ---
min_cpu = np.random.uniform(5, 20, num_records)
avg_cpu = min_cpu + np.random.uniform(5, 15, num_records)
max_cpu = avg_cpu + np.random.uniform(5, 20, num_records)

# --- Veriyi DataFrame'e Koyma ---
df = pd.DataFrame({
    'timestamp': timestamps,
    'min_cpu': min_cpu,
    'max_cpu': max_cpu,
    'avg_cpu': avg_cpu
})

# --- Bilerek Birkaç Anormallik Ekleme (Yüksek CPU Kullanımı) ---
# Rastgele 30 adet veri noktasını anormallik olarak işaretle
anomaly_indices = np.random.choice(df.index, size=30, replace=False)
for idx in anomaly_indices:
    df.loc[idx, 'avg_cpu'] = np.random.uniform(92, 99)
    df.loc[idx, 'min_cpu'] = df.loc[idx, 'avg_cpu'] - np.random.uniform(1, 5)
    df.loc[idx, 'max_cpu'] = 100.0

# Değerleri ondalık olarak yuvarla
df = df.round(2)

# --- CSV Dosyasına Kaydetme ---
output_path = "ornek_veriler/ornek_anormallik_verisi.csv"
df.to_csv(output_path, index=False)

print(f"Başarılı! '{output_path}' dosyası yeni formatla oluşturuldu.")
print("\nOluşturulan veriden birkaç örnek:")
print(df.sample(10).sort_values(by='timestamp'))
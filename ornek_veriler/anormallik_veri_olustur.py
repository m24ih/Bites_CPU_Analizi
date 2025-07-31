import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Anormallik tespiti için örnek veri seti oluşturuluyor...")

# Parametreler
num_records = 200
end_time = datetime.now()
start_time = end_time - timedelta(hours=num_records / 6) # 10 dakikalık aralıklar varsayımı

# Zaman damgaları
timestamps = pd.to_datetime(np.linspace(start_time.timestamp(), end_time.timestamp(), num_records), unit='s').round('s')

# Normal CPU verileri oluştur
min_cpu = np.random.uniform(5, 20, num_records)
avg_cpu = min_cpu + np.random.uniform(5, 15, num_records)
max_cpu = avg_cpu + np.random.uniform(5, 20, num_records)

# Veriyi DataFrame'e koy
df = pd.DataFrame({
    'timestamp': timestamps,
    'min_cpu': min_cpu,
    'max_cpu': max_cpu,
    'avg_cpu': avg_cpu
})

# Bilerek birkaç anormallik ekleyelim (yüksek CPU kullanımı)
anomaly_indices = np.random.choice(df.index, size=15, replace=False)
for idx in anomaly_indices:
    df.loc[idx, 'avg_cpu'] = np.random.uniform(92, 99)
    df.loc[idx, 'min_cpu'] = df.loc[idx, 'avg_cpu'] - np.random.uniform(1, 5)
    df.loc[idx, 'max_cpu'] = 100.0

df = df.round(2)

# CSV'ye kaydet
output_path = "ornek_veriler/ornek_anormallik_1.csv"
df.to_csv(output_path, index=False)

print(f"Başarılı! '{output_path}' dosyası oluşturuldu.")
print("\nOluşturulan veriden birkaç örnek (anormallikler karışık sırada):")
print(df.sample(10))
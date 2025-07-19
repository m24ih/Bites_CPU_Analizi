import pandas as pd
import os
import time

def process_all_files_for_average_load():
    """
    Tüm veri dosyalarını okur, her bir zaman adımı için tüm sanal makinelerin
    ortalama CPU kullanımını hesaplar ve sonucu tek bir CSV dosyasına kaydeder.
    Bu, genel sunucu yükünü temsil eder.
    """
    # --- Ayarlar ---
    DATA_PATH = 'data'
    # 1'den 20'ye kadar olan tüm dosyaları okuyacak şekilde ayarlandı.
    FILES_TO_READ = [f"vm_cpu_readings-file-{i}-of-195.csv.gz" for i in range(1, 196)]
    COLUMN_NAMES = ['timestamp', 'vmId', 'min_cpu', 'max_cpu', 'avg_cpu']
    CHUNK_SIZE = 1_000_000  # Belleği zorlamamak için her seferinde okunacak satır sayısı
    OUTPUT_CSV_PATH = 'genel_sunucu_yuku.csv'

    print("--- ADIM 1: Genel Sunucu Yükü Hesaplama Başladı ---")
    start_time = time.time()

    # Her dosyadan/parçadan elde edilen kısmi ortalamaları bu listede toplayacağız.
    partial_results = []

    for file_name in FILES_TO_READ:
        full_path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(full_path):
            print(f"- UYARI: '{file_name}' bulunamadı.")
            continue

        print(f"- İşleniyor: {file_name}")
        try:
            # Sadece gerekli sütunları okuyarak işlemi hızlandırıyoruz.
            chunk_iterator = pd.read_csv(
                full_path,
                header=None,
                names=COLUMN_NAMES,
                usecols=['timestamp', 'avg_cpu'],
                compression='gzip',
                chunksize=CHUNK_SIZE
            )

            for chunk in chunk_iterator:
                # Bu parçadaki her bir 'timestamp' için CPU ortalamasını hesapla.
                # Bu, henüz nihai ortalama değil, sadece bu parçanın ortalamasıdır.
                chunk_avg = chunk.groupby('timestamp')['avg_cpu'].mean()
                partial_results.append(chunk_avg)

        except Exception as e:
            print(f"  > HATA: {file_name} işlenirken bir sorun oluştu: {e}")

    if not partial_results:
        print("\nHiçbir veri işlenemedi. Lütfen dosya yollarını kontrol edin.")
        return

    print("\nTüm parçalardan elde edilen sonuçlar birleştiriliyor...")
    # Tüm kısmi ortalamaları birleştir.
    all_results = pd.concat(partial_results)
    
    # Bir timestamp farklı dosyalarda/parçalarda olabilir.
    # Bu yüzden, şimdi tüm ortalamaların ortalamasını alarak nihai sonucu buluyoruz.
    print("Nihai ortalamalar hesaplanıyor...")
    final_avg = all_results.groupby(all_results.index).mean()
    
    # Sonucu DataFrame'e çevirip zaman damgasına göre sırala
    final_df = final_avg.to_frame(name='ortalama_cpu_yuku').sort_index()
    
    # Oluşturulan temiz veriyi dosyaya kaydet
    print(f"Veri '{OUTPUT_CSV_PATH}' dosyasına kaydediliyor...")
    final_df.to_csv(OUTPUT_CSV_PATH)  # index (timestamp) de dosyaya yazılacak
    
    end_time = time.time()
    print("\n--- İŞLEM BAŞARIYLA TAMAMLANDI! ---")
    print(f"Toplam {len(final_df)} benzersiz zaman adımına ait genel yük hesaplandı.")
    print(f"Sonuç dosyası: {OUTPUT_CSV_PATH}")
    print(f"Toplam süre: {end_time - start_time:.2f} saniye.")

if __name__ == "__main__":
    process_all_files_for_average_load()
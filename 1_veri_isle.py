import pandas as pd
import os
import time

def process_all_files_for_all_metrics():
    """
    Tüm veri dosyalarını okur, her bir zaman adımı için tüm sanal makinelerin
    min_cpu, max_cpu ve avg_cpu değerlerinin ortalamasını hesaplar ve
    sonucu tek bir CSV dosyasına kaydeder.
    """

    DATA_PATH = 'data'
    FILES_TO_READ = [f"vm_cpu_readings-file-{i}-of-195.csv.gz" for i in range(1, 196)]
    COLUMN_NAMES = ['timestamp', 'vmId', 'min_cpu', 'max_cpu', 'avg_cpu']
    USE_COLS = ['timestamp', 'min_cpu', 'max_cpu', 'avg_cpu']
    CHUNK_SIZE = 1_000_000
    OUTPUT_CSV_PATH = 'genel_sunucu_yuku_tum_metrikler.csv'

    print("--- ADIM 1: Tüm CPU Metrikleri İçin Genel Yük Hesaplama Başladı ---")
    start_time = time.time()
    
    partial_results = []

    for file_name in FILES_TO_READ:
        full_path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(full_path):
            continue

        print(f"- İşleniyor: {file_name}")
        try:
            chunk_iterator = pd.read_csv(
                full_path,
                header=None,
                names=COLUMN_NAMES,
                usecols=USE_COLS,
                compression='gzip',
                chunksize=CHUNK_SIZE
            )
            for chunk in chunk_iterator:
                # Bu parçadaki her bir 'timestamp' için tüm CPU metriklerinin ortalamasını hesapla
                chunk_avg = chunk.groupby('timestamp')[['min_cpu', 'max_cpu', 'avg_cpu']].mean()
                partial_results.append(chunk_avg)
        except Exception as e:
            print(f"  > HATA: {file_name} işlenirken bir sorun oluştu: {e}")

    if not partial_results:
        print("\nHiçbir veri işlenemedi. Lütfen 'data' klasöründe dosyaların olduğundan emin olun.")
        return None

    print("\nTüm parçalardan elde edilen sonuçlar birleştiriliyor...")
    all_results = pd.concat(partial_results)
    
    print("Nihai ortalamalar hesaplanıyor...")
    # Tüm ortalamaların ortalamasını alarak nihai sonucu bul
    final_df = all_results.groupby(all_results.index).mean()
    
    # Sütunları isteğinize göre yeniden adlandır
    final_df.columns = ['avg_min_cpu', 'avg_max_cpu', 'avg_avg_cpu']
    
    final_df.sort_index(inplace=True)
    
    print(f"Veri '{OUTPUT_CSV_PATH}' dosyasına kaydediliyor...")
    final_df.to_csv(OUTPUT_CSV_PATH)
    
    end_time = time.time()
    print("\n--- ADIM 1 BAŞARIYLA TAMAMLANDI! ---")
    print(f"Sonuç dosyası: {OUTPUT_CSV_PATH}")
    print(f"Toplam süre: {end_time - start_time:.2f} saniye.")
    return OUTPUT_CSV_PATH


if __name__ == "__main__":
    # Adım 1: Tüm ham veriyi işle ve tüm metrikleri içeren temiz bir CSV dosyası oluştur
    output_file = process_all_files_for_all_metrics()


# BİTES Staj Projesi: CPU Kullanım Analizi, Tahmini ve Tespiti

Bu proje, BİTES Savunma & Havacılık staj programı kapsamında **Melih AK** tarafından geliştirilmiştir. Projenin temel amacı, sunucu sistemlerindeki CPU kullanım verilerini analiz ederek, gelecekteki davranışları tahmin eden ve anlık anormal durumları tespit eden hibrit bir yapay zeka sistemi oluşturmaktır.

Proje, interaktif bir web arayüzü ile sunulmakta ve tüm bağımlılıklarıyla birlikte bir Docker konteyneri içinde paketlenmiştir.

---

## 🚀 Özellikler

Bu proje iki ana modülden oluşmaktadır:

1.  **Zaman Serisi Tahmin (Kestirim) Modülü:**
    * Sistemin genel CPU metriklerinin (ortalama min, max ve avg) gelecekteki değerlerini tahmin eder.
    * **LSTM**, **SARIMA** ve **Transformer** gibi farklı model mimarileri karşılaştırılmıştır.
    * Tüm model eğitim süreçleri ve sonuçları **Weights & Biases (WandB)** ile takip edilmiştir.
    * Yapılan deneyler sonucunda en başarılı modelin **LSTM** olduğu belirlenmiştir.

2.  **Anormal Durum Tespit (Tespit) Modülü:**
    * Bireysel sunucu okumalarındaki anlık aşırı CPU kullanımı (%90 üzeri) durumlarını sınıflandırır.
    * **Decision Tree** ve **Random Forest** modelleri karşılaştırılmıştır.
    * Anormal durumları yakalamada daha başarılı olan **Random Forest** modeli nihai model olarak seçilmiştir.

3.  **İnteraktif Web Arayüzü:**
    * **Streamlit** kullanılarak geliştirilen arayüz, kullanıcıların her iki modülü de canlı olarak test etmesine olanak tanır.
    * Kullanıcılar, tahmin modülü için kendi veri setlerini yükleyebilir veya tespit modülü için manuel değerler girerek anlık testler yapabilir.

---

## 📂 Proje Yapısı

Proje, yönetimi kolaylaştırmak için modüler bir klasör yapısına sahiptir:

```
.
├── Dockerfile                      # Projeyi paketlemek için Docker tarifi
├── requirements.txt                # Gerekli Python kütüphaneleri
├── README.md                       # Bu dosya
├── 1_veri_isle.py                  # Ham veriyi kullanılabilecek şekilde işleyip kaydeder
├── 2_LSTM_Modeli.ipynb             # LSTM Modeli hakkındaki notebook
├── 2_SARIMA_Modeli.ipynb           # SARIMA Modeli hakkındaki notebook
├── 2_TRANSFORMER_Modeli.ipynb      # Transformer Modeli hakkındaki notebook
├── 2.1_LSTM_Wandb.ipynb            # LSTM Modelini Wandb platformuna loglamak için yapılan notebook
├── 2.1_SARIMA_Wandb.ipynb          # SARIMA Modelini Wandb platformuna loglamak için yapılan notebook
├── 2.1_TRANSFORMER_Wandb.iypnb     # Transformer Modelini Wandb platformuna loglamak için yapılan notebook
├── 3_Tespit_Modulu.ipynb           # İçerisinde Decision Tree ve Random Forest olan Tespit modülü notebook'u
├── 3.1_Tespit_Wandb.ipynb          # Tespit modülünü Wandb platformuna loglamak için yapılan notebook
├── 4_app.py                        # Streamlit web uygulamasının ana kodu 
├── 📁 EDA/                         # Veriseti hakkında ön bilgi edinmek için oluşturulan notebooklar
├── 📁 ornek_veriler/               # İşlenmiş veri setleri
└── 📁 model_path/                  # Eğitilmiş ve kaydedilmiş en iyi modeller (.pt, .joblib)
```

---

## 🚀 Canlı Demo (Live Demo)

Projeyi yerel olarak kurmak yerine, aşağıdaki link üzerinden canlı olarak test edebilirsiniz:

### 👉 [https://bites-staj.melihak.me](https://bites-staj.melihak.me)

---

## 🛠️ Kurulum ve Çalıştırma (Docker ile)

Bu proje, tüm bağımlılıklarıyla birlikte bir Docker konteyneri içinde çalışacak şekilde tasarlanmıştır. Projeyi çalıştırmak için bilgisayarınızda **Git** ve **Docker Desktop**'ın kurulu olması yeterlidir.

### Adım 1: Projeyi Klonlayın

```bash
git clone https://github.com/m24ih/Bites_CPU_Analizi.git
cd Bites_CPU_Analizi
```

### Adım 2: Docker İmajını Oluşturun

Proje ana klasöründeyken aşağıdaki komutu çalıştırarak projenin Docker imajını oluşturun. Bu işlem, kütüphaneler indirileceği için ilk seferde birkaç dakika sürebilir.

```bash
docker build -t bites_staj_projesi .
```

### Adım 3: Docker Konteynerini Çalıştırın

İmaj oluşturulduktan sonra, aşağıdaki komutla uygulamayı çalıştırın:

```bash
docker run -p 8501:8501 --name bites_app bites_staj_projesi
```

### Adım 4: Uygulamaya Erişin

Uygulama başarıyla başladığında, web tarayıcınızı açın ve aşağıdaki adrese gidin:

**http://localhost:8501**

Artık projenin interaktif arayüzünü kullanabilirsiniz.

---

## 💻 Kullanılan Teknolojiler

* **Veri Analizi ve İşleme:** Pandas, NumPy
* **Makine Öğrenmesi & Derin Öğrenme:** Scikit-learn, Statsmodels, Pmdarima, PyTorch
* **Deney Takibi:** Weights & Biases (WandB)
* **Web Arayüzü:** Streamlit
* **Paketleme ve Dağıtım:** Docker
* **Görselleştirme:** Plotly, Matplotlib
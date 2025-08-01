import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import math

# --- Metrikler için importlar ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# -------------------------------------------------------------------
# MODEL MİMARİLERİ VE YARDIMCI FONKSİYONLAR
# -------------------------------------------------------------------

# --- Tahmin Modeli (LSTM) Mimarisi ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, num_layers=2, dropout=0.2, output_size=3):
        """
        Belirtilen hiperparametrelerle modelin katmanlarını (LSTM ve Linear) tanımlar ve başlatır.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_layer_size,
            num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        """
        Girdi dizisini LSTM katmanından geçirir ve son zaman adımının çıktısını kullanarak nihai tahmini üretir.
        """
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# --- Transformer için Konumsal Kodlama (Positional Encoding) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Modelin pozisyon bilgisini anlaması için sinüs/kosinüs kodlama matrisini oluşturur ve bir dropout katmanı başlatır.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Girdi tensörüne pozisyonel kodlama değerlerini ekler ve ardından ezberlemeyi önlemek için dropout uygular.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- Tahmin Modeli (Transformer) Mimarisi ---
class TransformerModel(nn.Module):
    def __init__(self, input_size=3, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1, output_size=3):
        """
        Belirtilen hiperparametrelerle Transformer modelinin katmanlarını (girdi, pozisyonel kodlama, enkoder ve çıktı) tanımlar.
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        """
        Girdi dizisini model katmanlarından geçirir ve son zaman adımının çıktısını kullanarak nihai tahmini üretir.
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

# --- Modelleri ve Veriyi Yüklemek için Fonksiyonlar ---
@st.cache_resource
def load_forecasting_model(model_path, config):
    """
    Belirtilen yoldan eğitilmiş bir LSTM modelinin ağırlıklarını yükler ve modeli değerlendirme modunda döndürür.
    """
    device = torch.device('cpu')
    model = LSTMModel(
        hidden_layer_size=config['hidden_dim'], num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        return None

@st.cache_resource
def load_transformer_model(model_path, config):
    """
    Belirtilen yoldan eğitilmiş bir Transformer modelinin ağırlıklarını yükler ve modeli değerlendirme modunda döndürür.
    """
    device = torch.device('cpu')
    model = TransformerModel(
        d_model=config['d_model'], nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'], # Bu satır eklendi
        dropout=config['dropout']
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        return None

@st.cache_resource
def load_detection_model(model_path):
    """
    Belirtilen yoldan 'joblib' ile kaydedilmiş bir anormallik tespit modelini (örn: Random Forest) yükler.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def load_data(file_source):
    """
    Verilen bir kaynaktan CSV verisini yükler, sütunları yeniden adlandırır, 
    gerekli kontrolleri yapar ve zaman damgasını index olarak ayarlar.
    """
    try:
        df = pd.read_csv(file_source)
        rename_map = {'avg_min_cpu': 'min_cpu', 'avg_max_cpu': 'max_cpu', 'avg_avg_cpu': 'avg_cpu'}
        df.rename(columns=rename_map, inplace=True)
        required_cols = {'min_cpu', 'max_cpu', 'avg_cpu'}
        if not required_cols.issubset(df.columns):
            st.error(f"HATA: Gerekli sütunlar ('min_cpu', 'max_cpu', 'avg_cpu') bulunamadı.")
            return None
        if 'timestamp' not in df.columns:
            st.error("HATA: Yüklenen dosyada 'timestamp' sütunu bulunamadı.")
            return None
        if pd.api.types.is_numeric_dtype(df['timestamp']):
             df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
             df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df
    except FileNotFoundError:
        st.error(f"HATA: '{file_source}' adlı varsayılan veri dosyası bulunamadı.")
        return None
    except Exception as e:
        st.error(f"Veri yüklenirken bir hata oluştu: {e}")
        return None
    
def create_sequences(data, lookback_window):
    """
    Bir zaman serisi dizisini, modelin anlayacağı girdi (X) ve hedef (y) dizileri haline getirir.
    """
    X, y = [], []
    for i in range(len(data) - lookback_window):
        feature = data[i:(i + lookback_window)]
        target = data[i + lookback_window]
        X.append(feature)
        y.append(target)
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

# -------------------------------------------------------------------
# STREAMLIT ARAYÜZÜ
# -------------------------------------------------------------------

st.set_page_config(page_title="CPU Analiz Projesi", layout="wide")

st.sidebar.title("BİTES Staj Projesi")
st.sidebar.markdown("Melih AK")
st.sidebar.divider()
st.sidebar.title("Navigasyon")
page = st.sidebar.radio("Sayfayı Seçin", ["Ana Sayfa", "Zaman Serisi Tahmini", "Anormallik Tespiti"])

if page == "Ana Sayfa":
    st.title("CPU Kullanım Analizi, Tahmini ve Tespiti")
    st.markdown("Bu interaktif web uygulaması, staj projesi kapsamında geliştirilen makine öğrenmesi modellerini sergilemektedir.")
    st.markdown("""
    Proje iki ana modülden oluşmaktadır:
    - **Zaman Serisi Tahmini:** Sistemin genel CPU metriklerinin gelecekteki davranışını **LSTM** ve **Transformer** modelleri ile tahmin eder.
    - **Anormallik Tespiti:** Bireysel sunucu okumalarındaki anlık aşırı CPU kullanımını Random Forest modeli ile sınıflandırır.
    
    Sol taraftaki menüyü kullanarak modülleri test edebilirsiniz.
    """)
    st.info("Bu arayüz, Proje İster Dokümanı'nda belirtilen gereksinimleri karşılamak üzere Streamlit kullanılarak geliştirilmiştir.")

# --- Zaman Serisi Tahmini Sayfası ---
elif page == "Zaman Serisi Tahmini":
    st.header("Zaman Serisi Tahmin Modülü")

    # Ortak tahmin ve görselleştirme fonksiyonu
    def run_prediction_pipeline(model, df, model_name):
        """
        Verilen bir model ve veri seti ile tahminler oluşturur, sonuçları Streamlit arayüzünde bir grafik ve performans metrikleri olarak görselleştirir.
        """
        with st.spinner(f"{model_name} modeli ile tahminler ve metrikler hesaplanıyor..."):
            target_columns = ['min_cpu', 'max_cpu', 'avg_cpu']
            ts_data = df[target_columns].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            ts_data_scaled = scaler.fit_transform(ts_data)
            train_size = int(len(ts_data_scaled) * 0.8)
            lookback = 12
            X_test, y_test = create_sequences(ts_data_scaled[train_size:], lookback)
            
            with torch.no_grad():
                predictions_scaled = model(X_test).numpy()
            
            predictions_original_scale = scaler.inverse_transform(predictions_scaled)
            y_test_original_scale = scaler.inverse_transform(y_test.numpy())
            
            st.subheader(f"{model_name} Tahmin Sonuçları Grafiği")
            fig = go.Figure()
            plot_index = df.index[train_size+lookback:]
            for i, column in enumerate(target_columns):
                fig.add_trace(go.Scatter(x=plot_index, y=y_test_original_scale[:, i], mode='lines', name=f'Gerçek {column}'))
                fig.add_trace(go.Scatter(x=plot_index, y=predictions_original_scale[:, i], mode='lines', name=f'Tahmin {column}', line=dict(dash='dash')))
            fig.update_layout(title=f"CPU Metrikleri: Gerçek Değerler vs. {model_name} Tahminleri", xaxis_title="Zaman Damgası", yaxis_title="CPU Kullanımı (%)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"{model_name} Model Performans Metrikleri")
            metrics_data = {}
            for i, col_name in enumerate(target_columns):
                true_vals, pred_vals = y_test_original_scale[:, i], predictions_original_scale[:, i]
                mae, rmse, r2 = mean_absolute_error(true_vals, pred_vals), np.sqrt(mean_squared_error(true_vals, pred_vals)), r2_score(true_vals, pred_vals)
                non_zero_mask = true_vals != 0
                mape = mean_absolute_percentage_error(true_vals[non_zero_mask], pred_vals[non_zero_mask]) if np.any(non_zero_mask) else 0
                metrics_data[col_name] = {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}

            cols = st.columns(len(target_columns))
            for i, col_name in enumerate(target_columns):
                with cols[i]:
                    st.markdown(f"**--- Metrik: `{col_name}` ---**")
                    st.markdown(f"MAE: **{metrics_data[col_name]['MAE']:.2f}**")
                    st.markdown(f"RMSE: **{metrics_data[col_name]['RMSE']:.2f}**")
                    st.markdown(f"R-Kare (R²): **{metrics_data[col_name]['R²']:.2f}**")
                    st.markdown(f"MAPE: **{metrics_data[col_name]['MAPE']:.2%}**")
            st.success(f"{model_name} ile işlem başarıyla tamamlandı!")

    lstm_tab, transformer_tab = st.tabs(["LSTM Modeli", "Transformer Modeli"])

    with lstm_tab:
        st.subheader("LSTM Modeli ile Tahmin")
        lstm_config = {'hidden_dim': 50, 'num_layers': 1, 'dropout': 0.1}
        LSTM_PATH = "model_path/lstm_LSTM_h50_l1.pt"
        DEFAULT_DATA_PATH = "ornek_veriler/genel_sunucu_yuku_tum_metrikler.csv"
        
        lstm_model = load_forecasting_model(LSTM_PATH, lstm_config)
        st.divider()
        st.info("Analiz ve tahmin için bir CSV dosyası yükleyebilir veya varsayılan veri setini kullanabilirsiniz.")
        uploaded_file_lstm = st.file_uploader("LSTM için CSV yükle", type=['csv'], key="lstm_uploader")
        
        data_source_lstm = uploaded_file_lstm if uploaded_file_lstm is not None else DEFAULT_DATA_PATH
        df_lstm = load_data(data_source_lstm)

        if lstm_model is None:
            st.error(f"HATA: Gerekli LSTM modeli (`{LSTM_PATH}`) dosyası bulunamadı.")
        elif df_lstm is None:
            st.warning("Veri yüklenemediği için işleme devam edilemiyor.")
        else:
            if st.button("LSTM ile Tahminleri Oluştur"):
                run_prediction_pipeline(lstm_model, df_lstm, "LSTM")

    with transformer_tab:
        st.subheader("Transformer Modeli ile Tahmin")
        transformer_config = {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 2,
            'dim_feedforward': 128, # Hata mesajındaki modele göre bu değer 128 olmalı
            'dropout': 0.1
        }
        TRANSFORMER_PATH = "model_path/transformer_mild-planet-32.pt"
        DEFAULT_DATA_PATH_T = "ornek_veriler/genel_sunucu_yuku_tum_metrikler.csv"

        transformer_model = load_transformer_model(TRANSFORMER_PATH, transformer_config)
        st.divider()
        st.info("Analiz ve tahmin için bir CSV dosyası yükleyebilir veya varsayılan veri setini kullanabilirsiniz.")
        uploaded_file_transformer = st.file_uploader("Transformer için CSV yükle", type=['csv'], key="transformer_uploader")

        data_source_transformer = uploaded_file_transformer if uploaded_file_transformer is not None else DEFAULT_DATA_PATH_T
        df_transformer = load_data(data_source_transformer)

        if transformer_model is None:
            st.error(f"HATA: Gerekli Transformer modeli (`{TRANSFORMER_PATH}`) dosyası bulunamadı. Lütfen önce modeli eğitip bu yola kaydedin.")
        elif df_transformer is None:
            st.warning("Veri yüklenemediği için işleme devam edilemiyor.")
        else:
            if st.button("Transformer ile Tahminleri Oluştur"):
                run_prediction_pipeline(transformer_model, df_transformer, "Transformer")

# --- Anormallik Tespiti Sayfası ---
elif page == "Anormallik Tespiti":
    st.header("Anormallik Tespit Modülü (Random Forest)")
    
    DETECTOR_PATH = "model_path/RandomForest_serene-grass-8.joblib"
    DEFAULT_ANOMALY_DATA_PATH = "ornek_veriler/ornek_anormallik_verisi.csv"
    
    detection_model = load_detection_model(DETECTOR_PATH)

    if detection_model is None:
        st.error(f"HATA: Tespit modeli (`{DETECTOR_PATH}`) dosyası bulunamadı.")
    else:
        st.info("Modelin beklediği `min_cpu`, `max_cpu`, `avg_cpu` sütunlarını içeren bir CSV dosyası yükleyebilir veya varsayılan veri setini kullanabilirsiniz.")
        uploaded_file_detection = st.file_uploader("Anormallik tespiti için bir CSV dosyası yükleyin", type=['csv'], key="detection_uploader")

        # Veri kaynağını seç
        data_source_anomaly = uploaded_file_detection if uploaded_file_detection is not None else DEFAULT_ANOMALY_DATA_PATH

        if st.button("Anormallik Tespiti Yap"):
            try:
                with st.spinner("Dosyadaki veriler üzerinde anormallik tespiti yapılıyor..."):
                    df_detect = pd.read_csv(data_source_anomaly)
                    
                    rename_map = {'avg_min_cpu': 'min_cpu', 'avg_max_cpu': 'max_cpu', 'avg_avg_cpu': 'avg_cpu'}
                    df_detect.rename(columns=rename_map, inplace=True)
                    
                    required_cols = ['min_cpu', 'max_cpu', 'avg_cpu']
                    if all(col in df_detect.columns for col in required_cols):
                        data_to_predict = df_detect[required_cols]
                        predictions = detection_model.predict(data_to_predict)
                        probabilities = detection_model.predict_proba(data_to_predict)[:, 1]
                        df_detect['tespit_edilen_durum'] = np.where(predictions == 1, 'ANORMAL', 'NORMAL')
                        df_detect['anormal_olma_yuzdesi'] = probabilities * 100
                        anomalies = df_detect[df_detect['tespit_edilen_durum'] == 'ANORMAL'].copy()

                        st.subheader("Tespit Sonuçları")
                        if not anomalies.empty:
                            st.warning(f"Toplam {len(anomalies)} adet ANORMAL durum tespit edildi.")
                            display_cols = required_cols + ['anormal_olma_yuzdesi']
                            if 'timestamp' in anomalies.columns:
                                display_cols.insert(0, 'timestamp')
                            anomalies_display = anomalies[display_cols].reset_index(drop=True)
                            anomalies_display['anormal_olma_yuzdesi'] = anomalies_display['anormal_olma_yuzdesi'].map('{:.2f}%'.format)
                            st.dataframe(anomalies_display.style.highlight_max(axis=0, subset=['anormal_olma_yuzdesi'], color='lightcoral'), use_container_width=True)
                        else:
                            st.success("İncelenen veride herhangi bir ANORMAL durum tespit edilmedi.")
                    else:
                        st.error(f"HATA: Yüklediğiniz/varsayılan dosyada gerekli olan `{', '.join(required_cols)}` sütunlarından biri veya birkaçı eksik.")
            
            except FileNotFoundError:
                st.error(f"HATA: Varsayılan veri dosyası bulunamadı: '{DEFAULT_ANOMALY_DATA_PATH}'")
            except Exception as e:
                st.error(f"Dosya işlenirken bir hata oluştu: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# ------------------------------------------
# 1. SAYFA YAPILANDIRMASI
# ------------------------------------------
st.set_page_config(
    page_title="AI Meme Kanseri Teşhis Sistemi",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# 2. CSS STİLİ
# ------------------------------------------
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS dosyası bulunamadı. Varsayılan stiller kullanılıyor.")

local_css("style.css")

# ------------------------------------------
# 3. VERİ YÜKLEME VE MODEL EĞİTİMİ
# ------------------------------------------
@st.cache_resource(ttl=3600)
def load_and_train_models():
    # Verileri yükle
    df = pd.read_csv("data.csv")

    # Verileri temizle
    df['target'] = df['diagnosis'].map({'M': 0, 'B': 1})
    X = df.drop(['id', 'diagnosis', 'Unnamed: 32', 'target'], axis=1, errors='ignore')
    y = df['target']

    # Tren testi ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model eğitim fonksiyonu
    def train_model(model, name):
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy, train_time

    # Modeller
    ml_models = {
        'Lojistik Regresyon': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(kernel='rbf', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    }

    deep_learning_models = {}
    trained_models = {}

    # Makine Öğrenimi Modellerini Eğit
    for name, model in ml_models.items():
        trained_model, accuracy, train_time = train_model(model, name)
        trained_models[name] = {
            'model': trained_model,
            'accuracy': accuracy,
            'train_time': train_time,
            'type': 'ML'
        }

    # Yapay Sinir Ağı
    try:
        nn_model = keras.Sequential([
            keras.layers.Dense(30, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(15, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        start_time = time.time()
        history = nn_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100, batch_size=32, verbose=0
        )
        train_time = time.time() - start_time

        y_pred = (nn_model.predict(X_test_scaled) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        trained_models['Yapay Sinir Ağı'] = {
            'model': nn_model,
            'accuracy': accuracy,
            'train_time': train_time,
            'history': history,
            'type': 'DL'
        }
    except Exception as e:
        st.error(f"YSA modeli oluşturulurken hata: {e}")

    # CNN Modeli
    try:
        input_shape_cnn = (6, 5, 1) # 6 satır, 5 sütun, 1 kanal (gri tonlamalı)
        num_features = X_train_scaled.shape[1] # 30 olmalı

        if num_features < (input_shape_cnn[0] * input_shape_cnn[1]):
            # 30'dan az özellik varsa sıfırlarla doldurun
            padding_needed = (input_shape_cnn[0] * input_shape_cnn[1]) - num_features
            X_train_cnn = np.pad(X_train_scaled, ((0,0), (0, padding_needed)), 'constant')
            X_test_cnn = np.pad(X_test_scaled, ((0,0), (0, padding_needed)), 'constant')
        elif num_features > (input_shape_cnn[0] * input_shape_cnn[1]):
            # 30'dan fazla özellik varsa kes
            X_train_cnn = X_train_scaled[:, :input_shape_cnn[0] * input_shape_cnn[1]]
            X_test_cnn = X_test_scaled[:, :input_shape_cnn[0] * input_shape_cnn[1]]
        else:
            X_train_cnn = X_train_scaled
            X_test_cnn = X_test_scaled

        X_train_cnn = X_train_cnn.reshape(-1, *input_shape_cnn)
        X_test_cnn = X_test_cnn.reshape(-1, *input_shape_cnn)

        cnn_model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_cnn),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        start_time = time.time()
        history_cnn = cnn_model.fit(
            X_train_cnn, y_train,
            validation_data=(X_test_cnn, y_test),
            epochs=50, batch_size=32, verbose=0
        )
        train_time_cnn = time.time() - start_time

        y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype(int)
        accuracy_cnn = accuracy_score(y_test, y_pred_cnn)

        trained_models['Evrişimli Sinir Ağı (CNN)'] = {
            'model': cnn_model,
            'accuracy': accuracy_cnn,
            'train_time': train_time_cnn,
            'history': history_cnn,
            'input_shape_cnn': input_shape_cnn,
            'type': 'DL'
        }
    except Exception as e:
        st.error(f"CNN modeli oluşturulurken hata: {e}")


    # Autoencoder Modeli
    try:
        input_dim = X_train_scaled.shape[1]
        encoding_dim = 15 # Sıkıştırılmış temsil boyutu

        # Encoder katmanları için açıkça bir Input tanımla
        encoder_input = keras.layers.Input(shape=(input_dim,), name='encoder_input')
        encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoder_input)
        encoder = keras.Model(inputs=encoder_input, outputs=encoded, name='encoder_model')

        # Decoder katmanları için açıkça bir Input tanımla
        decoder_input = keras.layers.Input(shape=(encoding_dim,), name='decoder_input')
        decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoder_input) # Normalized data için sigmoid kullanılabilir
        decoder = keras.Model(inputs=decoder_input, outputs=decoded, name='decoder_model')

        # Autoencoder modelini birleştirme
        # Encoder'ın çıktısını Decoder'ın girdisi olarak doğrudan ver
        autoencoder_output = decoder(encoder(encoder_input))
        autoencoder = keras.Model(inputs=encoder_input, outputs=autoencoder_output, name='autoencoder_model')
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        start_time_ae = time.time()
        history_ae = autoencoder.fit(
            X_train_scaled, X_train_scaled, # Giriş ve hedef aynı
            epochs=50, batch_size=32, shuffle=True, verbose=0, # Daha hızlı deneme için epoch azaltıldı
            validation_data=(X_test_scaled, X_test_scaled)
        )
        train_time_ae = time.time() - start_time_ae

        # Özellik çıkarımı için encoder'ı kullan
        X_train_encoded = encoder.predict(X_train_scaled)
        X_test_encoded = encoder.predict(X_test_scaled)

        # Kodlanmış özellikler üzerinde bir sınıflandırıcı eğit
        ae_classifier = LogisticRegression(max_iter=1000)
        ae_classifier.fit(X_train_encoded, y_train)

        y_pred_ae = ae_classifier.predict(X_test_encoded)
        accuracy_ae = accuracy_score(y_test, y_pred_ae)

        trained_models['Autoencoder'] = {
            'model': ae_classifier, # Tahmin için sınıflandırıcıyı sakla
            'accuracy': accuracy_ae,
            'train_time': train_time_ae,
            'autoencoder_history': history_ae,
            'encoder': encoder, # Yeni veri üzerinde tahmin yapmak için encoder'ı sakla
            'type': 'DL'
        }
    except Exception as e:
        st.error(f"Autoencoder modeli oluşturulurken hata: {e}")


    return df, X.columns.tolist(), scaler, trained_models

# ------------------------------------------
# 4. KULLANICI GİRİŞ BİLEŞENİ
# ------------------------------------------
def user_input_features(features, df):
    """Kullanıcı girdi bileşenlerini oluştur"""
    data = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            data[feature] = st.slider(
                label=feature.replace('_', ' ').title(),
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].mean()),
                step=0.01,
                help=f"{feature.replace('_', ' ').title()} için tipik değer aralığı: {df[feature].min():.2f} - {df[feature].max():.2f}"
            )
    return pd.DataFrame(data, index=[0])

# ------------------------------------------
# 5. ANA UYGULAMA FONKSİYONU
# ------------------------------------------

def main():
    # Verileri ve modelleri yükle
    df, feature_names, scaler, model_dict = load_and_train_models()

    # Oturum durumunu başlat
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None

    # Sayfa başlığı
    st.title('🤖 AI Destekli Meme Kanseri Teşhis Sistemi')
    st.markdown("""
    <div style="background-color:#eaf2f8;padding:20px;border-radius:10px;margin-bottom:20px;">
    <h4 style="color:#1a5276;margin-top:0;">📌 Sistem Hakkında</h4>
    <p>Bu uygulama Baykar Teknolojisi tarafından mezuniyet projesi olarak verilip "Milli Teknoloji Akademesi, Yapay Zeka Uzmanlık Programı" için geliştirilmiştir.
    Wisconsin Meme Kanseri veri setini kullanarak çeşitli makine öğrenmesi algoritmalarıyla
    kanserli hücrelerin iyi huylu (benign) veya kötü huylu (malignant) olduğunu tahmin eder.</p>
    </div>
    """, unsafe_allow_html=True)

    # Diğer bilgilendirme kısmı
    st.markdown("""
    <div style="margin-bottom:20px;padding:15px;background-color:#f0f2f6;border-radius:8px;">
    <p>Veri Seti: Breast Cancer Wisconsin (Diagnostic) Veri Seti — <a href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data" target="_blank">Kaggle'da görüntüle</a></p>
    <p><strong>İpek Zorpineci</strong> tarafından geliştirildi.</p>
    </div>
    """, unsafe_allow_html=True)

    # Kenar çubuğu
    with st.sidebar:
        st.header('⚙️ Kontrol Paneli')

        # Model tipi seçimi için Radio butonu
        model_type = st.radio(
            "Model Tipi Seçin:",
            ("🧠 Makine Öğrenimi", "💡 Derin Öğrenme"),
            index=0, # Varsayılan olarak Makine Öğrenimi seçili
            help="Tahmin için kullanmak istediğiniz model kategorisini seçin."
        )

        available_models = []
        if model_type == "🧠 Makine Öğrenimi":
            available_models = [name for name, info in model_dict.items() if info['type'] == 'ML']
        elif model_type == "💡 Derin Öğrenme":
            available_models = [name for name, info in model_dict.items() if info['type'] == 'DL']

        # Dinamik dropdown menü
        model_choice = st.selectbox(
            'Model Seçin',
            available_models,
            help="Seçtiğiniz kategoriye ait makine öğrenmesi veya derin öğrenme algoritmasını seçin"
        )

        st.subheader('🩺 Hasta Verileri')
        input_df = user_input_features(feature_names, df)

        if st.button('🔍 Tahmin Yap', use_container_width=True):
            if model_choice: # Model seçimi yapılmışsa devam et
                with st.spinner(f'{model_choice} modeli ile tahmin yapılıyor...'):
                    try:
                        model_info = model_dict[model_choice]
                        model = model_info['model']
                        input_scaled = scaler.transform(input_df)

                        start_time = time.time()
                        if model_choice == 'Yapay Sinir Ağı':
                            prediction = (model.predict(input_scaled) > 0.5).astype(int)[0][0]
                            proba = model.predict(input_scaled)[0][0]
                        elif model_choice == 'Evrişimli Sinir Ağı (CNN)':
                            input_shape_cnn = model_info['input_shape_cnn']
                            num_features = input_scaled.shape[1]

                            if num_features < (input_shape_cnn[0] * input_shape_cnn[1]):
                                padding_needed = (input_shape_cnn[0] * input_shape_cnn[1]) - num_features
                                input_cnn = np.pad(input_scaled, ((0,0), (0, padding_needed)), 'constant')
                            elif num_features > (input_shape_cnn[0] * input_shape_cnn[1]):
                                input_cnn = input_scaled[:, :input_shape_cnn[0] * input_shape_cnn[1]]
                            else:
                                input_cnn = input_scaled

                            input_cnn = input_cnn.reshape(-1, *input_shape_cnn)
                            prediction = (model.predict(input_cnn) > 0.5).astype(int)[0][0]
                            proba = model.predict(input_cnn)[0][0]
                        elif model_choice == 'Autoencoder':
                            encoder = model_info['encoder']
                            input_encoded = encoder.predict(input_scaled)
                            prediction = model.predict(input_encoded)[0]
                            proba = model.predict_proba(input_encoded)[0][1]
                        else:
                            prediction = model.predict(input_scaled)[0]
                            proba = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

                        process_time = time.time() - start_time

                        st.session_state.prediction_results = {
                            'prediction': prediction,
                            'proba': proba,
                            'model_info': model_info,
                            'process_time': process_time,
                            'model_name': model_choice
                        }
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Tahmin sırasında hata oluştu: {str(e)}")
            else:
                st.warning("Lütfen bir model tipi ve ardından bir model seçin.")


    # Ana sekmeler
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Genel Bilgiler", "📈 Veri Analizi Sonuçları", "📉 Model Performansları", "ℹ️ Sistem Bilgileri"])

    # 1. GENEL BİLGİLER sekmesi (sabit içerik)
    with tab1:
        st.header("📊 Veri Seti Genel Bilgileri")

        # Temel istatistikler
        with st.expander("📋 Temel İstatistikler"):
            st.dataframe(df.drop(['id', 'diagnosis', 'Unnamed: 32', 'target'], axis=1, errors='ignore')
                         .describe().style.format("{:.2f}"),
                         use_container_width=True)

        # Dağıtım grafikleri
        st.subheader("Veri Dağılımları")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Teşhis Dağılımı")
            fig = px.pie(df, names='diagnosis',
                         title='İyi/Kötü Huylu Dağılımı',
                         color='diagnosis',
                         color_discrete_map={'B': '#27ae60', 'M': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Özellik Dağılımları")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_feature = st.selectbox("Özellik Seçin", numeric_cols[2:10])  # Skip id and target
            fig = px.histogram(df, x=selected_feature, color='diagnosis',
                               barmode='overlay', marginal="box",
                               color_discrete_map={'B': '#27ae60', 'M': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

        # Korelasyon matrisi
        st.subheader("Özellik Korelasyonları")
        numeric_df = df.select_dtypes(include=[np.number]).drop(['id', 'target'], axis=1, errors='ignore')
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    # 2. TAHMİN SONUÇLARI sekmesi (tahmine özel içerik)
    with tab2:
        st.header("📈 Veri Analizi Sonuçları")

        if st.session_state.prediction_results:
            prediction = st.session_state.prediction_results['prediction']
            proba = st.session_state.prediction_results['proba']
            model_info = st.session_state.prediction_results['model_info']
            process_time = st.session_state.prediction_results['process_time']
            model_name = st.session_state.prediction_results['model_name']

            # Teşhis sonucu
            if prediction == 1:
                diagnosis = 'İyi Huylu (Benign)'
                color = "#27ae60"
                icon = "✅"
            else:
                diagnosis = 'Kötü Huylu (Malignant)'
                color = "#e74c3c"
                icon = "⚠️"

            st.markdown(f"""
            <div style="border-left:5px solid {color};padding:15px;background-color:#f8f9fa;border-radius:8px;margin-bottom:20px;">
                <h3 style="color:{color};margin:0;">{icon} {model_name} Modeli Sonucu: <span style="font-weight:bold;">{diagnosis}</span></h3>
            </div>
            """, unsafe_allow_html=True)

            # Güncel dağılım
            st.subheader("Güncel Teşhis Dağılımı (Tahmin Dahil)")
            temp_df = df.copy()
            new_row = pd.DataFrame({'diagnosis': ['B' if prediction == 1 else 'M']})
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)

            col1, col2 = st.columns(2)
            with col1:
                if proba is not None:
                    if prediction == 1:
                        st.metric(label="İyi Huylu Olma Olasılığı",
                                 value=f"{proba*100:.2f}%",
                                 delta=f"{proba*100-50:.2f}%")
                    else:
                        st.metric(label="Kötü Huylu Olma Olasılığı",
                                 value=f"{(1-proba)*100:.2f}%",
                                 delta=f"{(1-proba)*100-50:.2f}%")

                fig = px.pie(temp_df, names='diagnosis',
                             title='İyi/Kötü Huylu Dağılımı',
                             color='diagnosis',
                             color_discrete_map={'B': '#27ae60', 'M': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                diagnosis_counts = temp_df['diagnosis'].value_counts().reset_index()
                diagnosis_counts.columns = ['Teşhis', 'Sayı']
                diagnosis_counts['Teşhis'] = diagnosis_counts['Teşhis'].map({'B': 'İyi Huylu', 'M': 'Kötü Huylu'})
                st.dataframe(diagnosis_counts, hide_index=True, use_container_width=True)

            # Model detayları
            with st.expander("🧠 Model Detayları"):
                cols = st.columns(3)
                cols[0].metric("Test Doğruluğu", f"{model_info['accuracy']*100:.2f}%")
                cols[1].metric("Eğitim Süresi", f"{model_info['train_time']:.3f} sn")
                cols[2].metric("Tahmin Süresi", f"{process_time:.3f} sn")

                if model_name == 'Yapay Sinir Ağı' and 'history' in model_info:
                    st.subheader("Eğitim Geçmişi")
                    history = model_info['history']
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
                    ax1.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
                    ax1.set_title('Model Doğruluğu')
                    ax1.legend()
                    ax2.plot(history.history['loss'], label='Eğitim Kaybı')
                    ax2.plot(history.history['val_loss'], label='Doğrulama Kaybı')
                    ax2.set_title('Model Kaybı')
                    ax2.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                elif model_name == 'Evrişimli Sinir Ağı (CNN)' and 'history' in model_info:
                    st.subheader("CNN Eğitim Geçmişi")
                    history = model_info['history']
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
                    ax1.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
                    ax1.set_title('CNN Doğruluğu')
                    ax1.legend()
                    ax2.plot(history.history['loss'], label='Eğitim Kaybı')
                    ax2.plot(history.history['val_loss'], label='Doğrulama Kaybı')
                    ax2.set_title('CNN Kaybı')
                    ax2.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                elif model_name == 'Autoencoder' and 'autoencoder_history' in model_info:
                    st.subheader("Autoencoder Eğitim Geçmişi (Rekonstrüksiyon Kaybı)")
                    history = model_info['autoencoder_history']
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(history.history['loss'], label='Eğitim Kaybı')
                    ax.plot(history.history['val_loss'], label='Doğrulama Kaybı')
                    ax.set_title('Autoencoder Rekonstrüksiyon Kaybı')
                    ax.set_ylabel('Kayıp (MSE)')
                    ax.set_xlabel('Epoch')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

            # ------------------------------------------
            # 1. RADAR CHART (Özellik Profili)
            # ------------------------------------------
            st.markdown("---")
            st.subheader("📊 Özellik Profil Karşılaştırması")

            # Hasta verileri ve ortalama değerler
            mean_benign = df[df['diagnosis']=='B'][feature_names].mean()
            mean_malignant = df[df['diagnosis']=='M'][feature_names].mean()
            patient_values = input_df.iloc[0]

            # Radar chart için veri hazırlama
            categories = [f.replace('_', ' ').title() for f in feature_names]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=mean_benign.values,
                theta=categories,
                fill='toself',
                name='Ortalama İyi Huylu',
                line_color='green'
            ))

            fig.add_trace(go.Scatterpolar(
                r=mean_malignant.values,
                theta=categories,
                fill='toself',
                name='Ortalama Kötü Huylu',
                line_color='red'
            ))

            fig.add_trace(go.Scatterpolar(
                r=patient_values.values,
                theta=categories,
                name='Hasta Değerleri',
                line_color='blue'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(mean_malignant.max(), patient_values.max())*1.1]
                    )),
                showlegend=True,
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------
            # 2. ÖZELLİK ÖNEM SIRALAMASI
            # ------------------------------------------
            # Seçilen modele göre özellik önemlerini kontrol edin
            if model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                st.markdown("---")
                st.subheader("🧠 Model Karar Önem Sıralaması")

                # Bu modeller için model doğrudan 'model_info['model']' içinde saklanır
                importances = model_info['model'].feature_importances_
                indices = np.argsort(importances)[::-1]

                fig = px.bar(x=np.array(feature_names)[indices][:10],
                             y=importances[indices][:10],
                             labels={'x':'Özellikler', 'y':'Önem Skoru'},
                             color=importances[indices][:10],
                             color_continuous_scale='Blues',
                             title='Modelin Karar Vermede Kullandığı En Önemli 10 Özellik')

                st.plotly_chart(fig, use_container_width=True)
            elif model_name == 'Lojistik Regresyon':
                st.markdown("---")
                st.subheader("🧠 Lojistik Regresyon Katsayı Önem Sıralaması")
                # Lojistik Regresyon için katsayılar özellik önemini (mutlak değer) gösterir
                coef = model_info['model'].coef_[0]
                abs_coef = np.abs(coef)
                indices = np.argsort(abs_coef)[::-1]

                fig = px.bar(x=np.array(feature_names)[indices][:10],
                             y=abs_coef[indices][:10],
                             labels={'x':'Özellikler', 'y':'Katsayı Mutlak Değeri'},
                             color=abs_coef[indices][:10],
                             color_continuous_scale='Blues',
                             title='Lojistik Regresyon Modelinin Karar Vermede Kullandığı En Önemli 10 Özellik')
                st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------
            # 3. PCA/TSNE DAĞILIM GÖRSELLEŞTİRMESİ
            # ------------------------------------------
            st.markdown("---")
            st.subheader("📍 Hasta Verisinin Genel Dağılımdaki Konumu")

            # PCA ile boyut indirgeme
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(scaler.transform(df[feature_names]))
            patient_embedding = reducer.transform(scaler.transform(input_df))

            plot_df = pd.DataFrame(embedding, columns=['x', 'y'])
            plot_df['diagnosis'] = df['diagnosis']
            plot_df['size'] = 5
            patient_point = pd.DataFrame({
                'x': [patient_embedding[0,0]],
                'y': [patient_embedding[0,1]],
                'diagnosis': ['Hasta'],
                'size': [15]
            })

            fig = px.scatter(
                pd.concat([plot_df, patient_point]),
                x='x', y='y',
                color='diagnosis',
                size='size',
                color_discrete_map={'B': 'green', 'M': 'red', 'Hasta': 'blue'},
                title='Hasta Verisinin Genel Dağılımdaki Konumu (PCA)',
                hover_data={'x':False, 'y':False}
            )

            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------
            # 4. DİNAMİK BOXPLOT (ÖZELLİK DAĞILIMI)
            # ------------------------------------------
            st.markdown("---")
            st.subheader("📦 Hasta Değerlerinin Dağılım Karşılaştırması")

            selected_feature = st.selectbox("Dağılımını görmek istediğiniz özelliği seçin",
                                             feature_names, key='boxplot_feature')

            fig = px.box(df, y=selected_feature, color='diagnosis',
                         color_discrete_map={'B': 'green', 'M': 'red'},
                         title=f'{selected_feature.replace("_", " ").title()} Dağılımı')

            # Hasta değerini ekleme
            fig.add_hline(y=input_df[selected_feature].values[0],
                          line_dash="dot",
                          annotation_text="Hasta Değeri",
                          line_color="blue")

            st.plotly_chart(fig, use_container_width=True)


            # ------------------------------------------
            # 5. KARAR SINIRI GÖRSELLEŞTİRMESİ (2D)
            # ------------------------------------------
            st.markdown("---")
            st.subheader("🖍️ Model Karar Sınırları (2 Özellik)")

            col1, col2 = st.columns(2)
            with col1:
                feature_x = st.selectbox("X Ekseni Özelliği", feature_names, index=0, key='feature_x_dec_boundary')
            with col2:
                feature_y = st.selectbox("Y Ekseni Özelliği", feature_names, index=1, key='feature_y_dec_boundary')

            try:
                # Sadece seçilen iki özellik için bir alt veri çerçevesi oluştur
                X_2d_raw = df[[feature_x, feature_y]]
                y_2d = df['target']

                # Sadece bu iki özellik için YENİ BİR SCALER tanımla ve fit et
                # Bu, mevcut genel scaler'ın tüm 30 özelliği beklemesini engeller.
                scaler_2d = StandardScaler()
                X_2d_scaled = scaler_2d.fit_transform(X_2d_raw)

                # Bu 2 özellik üzerinde bir Logistic Regression modeli eğit
                model_2d_boundary = LogisticRegression(max_iter=1000)
                model_2d_boundary.fit(X_2d_scaled, y_2d)

                fig, ax = plt.subplots()

                # Veri noktalarını çiz
                ax.scatter(X_2d_scaled[y_2d == 0, 0], X_2d_scaled[y_2d == 0, 1],
                           c='red', label='Kötü Huylu', alpha=0.5)
                ax.scatter(X_2d_scaled[y_2d == 1, 0], X_2d_scaled[y_2d == 1, 1],
                           c='green', label='İyi Huylu', alpha=0.5)

                # Karar sınırını çiz
                x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
                y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))

                Z = model_2d_boundary.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')

                # Hasta verisini işaretle
                # Sadece seçilen iki özelliği al ve bu iki özellik için oluşturulan scaler ile ölçekle
                patient_2d_input_raw = input_df[[feature_x, feature_y]]
                patient_2d_scaled = scaler_2d.transform(patient_2d_input_raw)

                ax.scatter(patient_2d_scaled[:,0], patient_2d_scaled[:,1],
                           marker='X', s=200, c='black', label='Hasta')

                ax.set_xlabel(feature_x.replace("_", " ").title())
                ax.set_ylabel(feature_y.replace("_", " ").title())
                ax.legend()
                ax.set_title(f'Karar Sınırı ({feature_x.replace("_", " ").title()} vs {feature_y.replace("_", " ").title()})')
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.warning(f"Karar sınırını görselleştirirken bir hata oluştu: {e}")
                st.info("Bu görselleştirme için yalnızca iki özelliğe sahip ayrı bir model eğitildi. Bu, genel bir örnek amaçlıdır.")

        else:
            st.info("ℹ️ Tahmin sonuçlarını görmek için lütfen önce tahmin yapınız.")

    # 3. MODEL PERFORMANSLARI sekmesi
    with tab3:
        st.header("📉 Model Performans Karşılaştırması")

        # Performans ölçümleri tablosu
        performance_data = []
        for name, info in model_dict.items():
            performance_data.append({
                'Model': name,
                'Doğruluk': info['accuracy'],
                'Eğitim Süresi (sn)': info['train_time'],
                'Tip': info['type'] # Yeni eklenen 'type' bilgisi
            })

        performance_df = pd.DataFrame(performance_data).sort_values('Doğruluk', ascending=False)

        st.dataframe(
            performance_df.style.format({
                'Doğruluk': '{:.2%}',
                'Eğitim Süresi (sn)': '{:.3f}'
            }).background_gradient(cmap='Blues'),
            use_container_width=True
        )

        # Performans çizelgeleri
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Doğruluk Karşılaştırması")
            # Model tipine göre renklendirme eklendi
            fig = px.bar(performance_df, x='Model', y='Doğruluk', color='Tip',
                         text_auto='.2%', color_discrete_map={'ML': 'lightblue', 'DL': 'purple'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Eğitim Süreleri")
            # Model tipine göre renklendirme eklendi
            fig = px.bar(performance_df, x='Model', y='Eğitim Süresi (sn)',
                         color='Tip',
                         color_discrete_map={'ML': 'lightgreen', 'DL': 'orange'})
            st.plotly_chart(fig, use_container_width=True)

        # NN eğitim geçmişi
        if 'Yapay Sinir Ağı' in model_dict and 'history' in model_dict['Yapay Sinir Ağı']:
            st.subheader("Yapay Sinir Ağı Eğitim Geçmişi")
            history = model_dict['Yapay Sinir Ağı']['history']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
            ax1.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
            ax1.set_title('Model Doğruluğu')
            ax1.set_ylabel('Doğruluk')
            ax1.set_xlabel('Epoch')
            ax1.legend()

            ax2.plot(history.history['loss'], label='Eğitim Kaybı')
            ax2.plot(history.history['val_loss'], label='Doğrulama Kaybı')
            ax2.set_title('Model Kaybı')
            ax2.set_ylabel('Kayıp')
            ax2.set_xlabel('Epoch')
            ax2.legend()

            st.pyplot(fig)
            plt.close(fig)

        # CNN eğitim geçmişi
        if 'Evrişimli Sinir Ağı (CNN)' in model_dict and 'history' in model_dict['Evrişimli Sinir Ağı (CNN)']:
            st.subheader("Evrişimli Sinir Ağı (CNN) Eğitim Geçmişi")
            history = model_dict['Evrişimli Sinir Ağı (CNN)']['history']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
            ax1.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
            ax1.set_title('CNN Doğruluğu')
            ax1.set_ylabel('Doğruluk')
            ax1.set_xlabel('Epoch')
            ax1.legend()

            ax2.plot(history.history['loss'], label='Eğitim Kaybı')
            ax2.plot(history.history['val_loss'], label='Doğrulama Kaybı')
            ax2.set_title('CNN Kaybı')
            ax2.set_ylabel('Kayıp')
            ax2.set_xlabel('Epoch')
            ax2.legend()

            st.pyplot(fig)
            plt.close(fig)

        # Autoencoder eğitim geçmişi
        if 'Autoencoder' in model_dict and 'autoencoder_history' in model_dict['Autoencoder']:
            st.subheader("Autoencoder Eğitim Geçmişi (Rekonstrüksiyon Kaybı)")
            history = model_dict['Autoencoder']['autoencoder_history']
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history.history['loss'], label='Eğitim Kaybı')
            ax.plot(history.history['val_loss'], label='Doğrulama Kaybı')
            ax.set_title('Autoencoder Rekonstrüksiyon Kaybı')
            ax.set_ylabel('Kayıp (MSE)')
            ax.set_xlabel('Epoch')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)


        # PCA Analizi
        st.markdown("---")
        st.subheader("PCA ile Boyut İndirgeme")
        with st.expander("📊 PCA Analiz Detayları"):
            # PCA için verileri hazırlayın
            X = df.drop(['id', 'diagnosis', 'Unnamed: 32', 'target'], axis=1, errors='ignore')
            y = df['diagnosis']

            # Verileri standartlaştırın
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)


            # PCA'yı uygula
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df['Diagnosis'] = y.values

            # PCA'nın grafiğini çizin
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Diagnosis',
                             title='PCA: İlk 2 Temel Bileşen',
                             color_discrete_map={'B': '#27ae60', 'M': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

            # Açıklanan varyans
            st.markdown(f"**Açıklanan Varyans Oranı:**")
            st.markdown(f"- PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
            st.markdown(f"- PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")

        # Tanıya Göre Özelliklerin Kutu Grafiği Izgarası
        st.markdown("---")
        st.subheader("Özelliklerin Teşhise Göre Dağılımı")
        with st.expander("📦 Boxplot Analizi"):
            # Görüntülenecek özellikleri seçin
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in ['id', 'target', 'Unnamed: 32']]
            selected_features = st.multiselect(
                "Görselleştirmek istediğiniz özellikleri seçin",
                features,
                default=features[:5]
            )

            if selected_features:

                # Çizim için veri çerçevesini eritin
                melt_df = pd.melt(df, id_vars=['diagnosis'], value_vars=selected_features)

                # Kutu grafikleri oluşturun
                fig = px.box(melt_df, x='variable', y='value', color='diagnosis',
                             color_discrete_map={'B': '#27ae60', 'M': '#e74c3c'},
                             title='Özelliklerin Teşhise Göre Dağılımı')
                fig.update_layout(xaxis_title='Özellikler', yaxis_title='Değer')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Lütfen en az bir özellik seçin")

        # Etkileşimli KDE Grid
        st.markdown("---")
        st.subheader("Özellik Dağılımları (KDE)")
        with st.expander("📊 KDE Dağılım Analizi"):
            # KDE için özellikleri seçin
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in ['id', 'target', 'Unnamed: 32']]
            selected_kde_features = st.multiselect(
                "KDE için özellik seçin",
                features,
                default=features[:3]
            )

            if selected_kde_features:
                # KDE grafikleri oluştur
                fig, axes = plt.subplots(len(selected_kde_features), 1,
                                         figsize=(10, 5*len(selected_kde_features)))

                if len(selected_kde_features) == 1:
                    axes = [axes]  # Tek bir subplot için düzeltme

                for i, feature in enumerate(selected_kde_features):
                    sns.kdeplot(data=df, x=feature, hue='diagnosis',
                                palette={'B': '#27ae60', 'M': '#e74c3c'},
                                ax=axes[i])
                    axes[i].set_title(f'{feature.replace("_", " ").title()} Dağılımı')
                    axes[i].set_xlabel('')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("Lütfen en az bir özellik seçin")


    # 4. SİSTEM BİLGİLERİ sekmesi
    with tab4:
        st.header("ℹ️ Sistem Bilgileri")
        st.markdown("""
        ### 🔧 Kullanılan Teknolojiler
        - 🐍 **Python**: 3.9+
        - 🖥️ **Streamlit**: Web arayüzü
        - 🤖 **Scikit-learn**: Geleneksel ML modelleri
        - 🧠 **TensorFlow/Keras**: Derin öğrenme modeli
        - 📊 **Plotly/Matplotlib**: Görselleştirmeler

        ### 📚 Veri Seti
        **Wisconsin Meme Kanseri Veri Seti**:
        - 🔢 Örnek Sayısı: 569
        - 📈 Özellikler: 30 sayısal özellik
        - 🎯 Hedef Değişken: İyi huylu (B) / Kötü huylu (M)

        ### ⚠️ Önemli Uyarı
        Bu sistem bir **karar destek aracıdır**. Kesin teşhis için mutlaka bir uzman hekime başvurunuz.
        """)


if __name__ == '__main__':
    main()
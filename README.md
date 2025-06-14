🩺 Meme Kanseri Sınıflandırma Panosu
Bu interaktif web uygulaması, Baykar Milli Teknoloji Akademisi - Yapay Zeka Uzmanlık Programı bitirme projesi kapsamında geliştirilmiştir. Kullanıcıların Breast Cancer Wisconsin (Diagnostic) Veri Seti'ni keşfetmelerine, çeşitli veri ön işleme tekniklerini uygulamalarına, makine öğrenimi ve derin öğrenme modellerini karşılaştırmalarına ve özel hasta girişleri üzerinde gerçek zamanlı tahminler yapmalarına olanak tanır.

📚 Bu Proje Şunları İçerir:
- Tamamen interaktif bir Streamlit Web Uygulaması (streamlit_app.py)
- Bağımsız analizler ve model geliştirmeleri için bir Jupyter Notebook (Baykar Projesi - İpek Zorpineci.ipynb)
- Metodolojiyi ve sonuçları özetleyen resmi bir Proje Raporu (BAYKAR(Yapay Zeka Analizi) Raporu.pdf)
- 
🧪 Veri Seti
Kaynak: Kaggle - Breast Cancer Wisconsin (Diagnostic) Data Set
İçerik: Meme kitlelerinin ince iğne aspiratlarının dijitalize edilmiş görüntülerinden çıkarılan 30 sayısal özellik ve 569 örnek.
Etiketler:
M = Malign (Kötü Huylu) (Kodlama sonrası 1'e dönüştürülmüştür)
B = Benign (İyi Huylu) (Kodlama sonrası 0'a dönüştürülmüştür)

🔍 Proje Akışı ve Özellikler
📊 1. Veri Yükleme ve Genel Bakış
data.csv dosyasının yüklenmesi ve ilk satırlarının, boyutunun, bilgi özetinin ve istatistiksel tanımlarının incelenmesi.
Gereksiz 'id' ve boş 'Unnamed: 32' sütunlarının kaldırılması.
Hedef değişken 'diagnosis'ın 'M': 1, 'B': 0 olacak şekilde sayısal hale dönüştürülmesi.

📈 2. Veri Analizi ve Görselleştirme
diagnosis dağılımının bar grafiği ile görselleştirilmesi.
Sayısal özelliklerin dağılımlarının (histogramlar ve KDE - Yoğunluk Tahmin Grafikleri) ve diagnosis'a göre farklılaşmalarının incelenmesi.
Tüm sayısal özellikler arasındaki korelasyonların ısı haritası ile görselleştirilmesi.

⚙️ 3. Veri Ön İşleme
Bağımlı ve bağımsız değişkenlerin ayrılması.
StandardScaler kullanılarak özelliklerin ölçeklendirilmesi (Standardizasyon).
Veri setinin eğitim ve test kümelerine (test_size=0.2, stratify=y ile) ayrılması.

🧠 4. Makine Öğrenimi Modelleri
Aşağıdaki geleneksel makine öğrenimi modelleri uygulanmış, eğitilmiş ve performansları karşılaştırılmıştır:

Lojistik Regresyon
Random Forest (Rastgele Orman)
Destek Vektör Makineleri (SVM)
K En Yakın Komşu (KNN)
Gaussian Naive Bayes
Gradient Boosting (Gradyan Artırma)
XGBoost

💡 5. Derin Öğrenme Modelleri
TensorFlow/Keras kullanılarak çeşitli derin öğrenme mimarileri uygulanmış, eğitilmiş ve performansları değerlendirilmiştir:

Yapay Sinir Ağı (YSA): Katmanlı ve Dropout içeren temel bir YSA.
Evrişimli Sinir Ağı (CNN): Tablusal veriye uyarlanmış deneysel bir CNN mimarisi.
Autoencoder: Boyut indirgeme amacıyla eğitilmiş, ardından öğrenilen özellikler üzerinde Lojistik Regresyon ile sınıflandırma yapılmıştır.

🌐 6. Streamlit Uygulaması (streamlit_app.py)
Projenin kullanıcı dostu web arayüzünü oluşturan ana uygulamadır. Şunları içerir:

Kullanıcı dostu hasta veri giriş arayüzü (slider'lar ile 30 özellik girişi).
Makine Öğrenimi ve Derin Öğrenme modelleri arasında seçim yapabilme.
Seçilen modelle gerçek zamanlı tahmin ve güven puanı görüntüleme.
Kapsamlı veri analizi ve model performans görselleştirmeleri sunan sekmeler:
Genel Bilgiler: Veri seti istatistikleri, teşhis dağılımı, özellik dağılımları ve korelasyon ısı haritaları.
Veri Analizi Sonuçları: Yapılan tahminin detayları, radar grafiği ile hasta özelliklerinin ortalama profillerle karşılaştırılması, özellik önem sıralaması, PCA ile veri dağılımında hastanın konumu, dinamik kutu grafikleri ve karar sınırı görselleştirmeleri.
Model Performansları: Tüm modellerin doğruluk oranları ve eğitim sürelerinin karşılaştırmalı tabloları ve grafikleri, derin öğrenme modellerinin eğitim geçmişi.
Sistem Bilgileri: Kullanılan teknolojiler ve veri seti hakkında genel bilgiler.

🛠️ Kurulum Talimatları
Bu projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

Gereksinimler
Python 3.9+ (veya projenizin kullandığı spesifik versiyon)
Temel Python kütüphaneleri (pandas, numpy)
Veri Bilimi ve ML Kütüphaneleri (scikit-learn, tensorflow, xgboost, plotly, seaborn, matplotlib)
Web Uygulaması Kütüphanesi (streamlit)
Kurulum
Depoyu Klonlayın:

Bash

git clone https://github.com/ipekZorpineci2003/Yapay_Zeka_Analizi.git
Proje Dizine Gidin:

Bash

cd Yapay_Zeka_Analizi
Gerekli Kütüphaneleri Yükleyin:
Gerekli kütüphaneleri içeren bir requirements.txt dosyanız yoksa (tavsiye edilir, aksi takdirde manuel olarak yüklemeniz gerekir):

(Eğer requirements.txt dosyanız yoksa, bu kısmı atlayın ve manuel yükleme yapın. Eğer varsa, içeriğinin güncel olduğundan emin olun.)

Bash

pip install -r requirements.txt
(Eğer requirements.txt dosyanız yoksa, aşağıdaki komutlarla temel kütüphaneleri yükleyebilirsiniz):

Bash

pip install pandas numpy scikit-learn tensorflow streamlit plotly xgboost matplotlib seaborn
Streamlit Uygulamasını Başlatın:

Bash

streamlit run streamlit_app.py
Uygulama otomatik olarak web tarayıcınızda açılacaktır.


👨‍💻 Yazar
Geliştiren: İpek Zorpineci
İletişim: LinkedIn Profiliniz • E-posta Adresiniz

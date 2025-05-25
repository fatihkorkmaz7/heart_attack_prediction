# 🫀 Heart Attack Prediction - ML Model with Kaggle Dataset

## 📌 Kaggle Notebook Linki
🔗 [Heart Attack Prediction - Kaggle Notebook](https://www.kaggle.com/code/fatihkorkmaz7/heart-attack-prediction?scriptVersionId=241832224)

---


## 📚 İçindekiler

- [📊 Proje Özeti](#-proje-özeti)
- [🧠 Kullanılan Makine Öğrenimi Yöntemleri](#-kullanılan-makine-öğrenimi-yöntemleri)
  - [🔹 Gözetimli Öğrenme Algoritmaları](#-gözetimli-öğrenme-algoritmaları-supervised-learning)
  - [🔹 Gözetimsiz Öğrenme Yöntemleri](#-gözetimsiz-öğrenme-yöntemleri-unsupervised-learning)
- [🔍 Problem Tanımı](#-problem-tanımı)
- [📈 Değerlendirme Metrikleri](#-değerlendirme-metrikleri)
- [🏥 Gerçek Hayatta Kullanımı](#-gerçek-hayatta-kullanımı)
- [🚀 Gelecekteki Geliştirme Fikirleri](#-gelecekteki-geliştirme-fikirleri)
- [🤝 Katkıda Bulunanlar](#-katkıda-bulunanlar)

---

## 📊 Proje Özeti

Bu projede, bireylerin kişisel sağlık verilerine dayanarak kalp krizi riskini tahmin eden bir **sınıflandırma modeli** geliştirildi. Ayrıca, **gözetimsiz öğrenme yöntemleri** ile veri kümesindeki örüntüler analiz edildi. 

Kullanılan veri seti Kaggle’dan alınmış, eksik değerler temizlenmiş ve görselleştirme, öznitelik mühendisliği, modelleme adımlarıyla detaylı şekilde işlenmiştir.

---

## 🧠 Kullanılan Makine Öğrenimi Yöntemleri

### 🔹 Gözetimli Öğrenme Algoritmaları (Supervised Learning)

Proje boyunca aşağıdaki dört gözetimli öğrenme algoritması uygulanmıştır:

1. **Logistic Regression**  
2. **k-Nearest Neighbors (KNN)**  
3. **Random Forest Classifier**  
4. **LightGBM Classifier** ✅ *En başarılı model*

Her bir modelin performansı 5-fold **cross-validation** yöntemiyle değerlendirilmiş, en yüksek doğruluk ve ROC-AUC skorunu sağlayan **LightGBM**, final model olarak seçilmiştir.

#### Hiperparametre Optimizasyonu:
LightGBM için `RandomizedSearchCV` ile:
- `num_leaves`
- `max_depth`
- `learning_rate`
- `n_estimators` gibi parametreler optimize edilmiştir.

---

### 🔹 Gözetimsiz Öğrenme Yöntemleri (Unsupervised Learning)

Proje kapsamında ayrıca veri kümesindeki gizli desenleri keşfetmek adına aşağıdaki gözetimsiz yöntemler uygulanmıştır:

1. **Principal Component Analysis (PCA)**  
   - Veri setinin boyutunu azaltmak ve görselleştirmeyi kolaylaştırmak için kullanıldı.
   - İlk iki bileşenle verinin %95'ten fazlası temsil edildi.

2. **K-Means Clustering**  
   - PCA sonrası veriler 3 farklı kümeye ayrıldı.
   - Her küme farklı birey gruplarını temsil etti ve kalp hastalığı görülme oranlarına göre yorumlandı.
   - Küme sayısı belirlenirken **Elbow Method** kullanıldı.

Bu gözetimsiz yöntemler, verinin yapısını daha iyi anlamayı ve farklı risk gruplarını keşfetmeyi sağlamıştır.

---

## 🔍 Problem Tanımı

Veri seti, yaş, cinsiyet, fiziksel aktivite, mental sağlık, sigara kullanımı, obezite ve diyabet gibi değişkenleri içermektedir. Hedef değişken: **HeartDisease (1 = Var, 0 = Yok)**

Amaç: Sağlık göstergelerine göre kalp krizi riskini doğru tahmin etmektir.

---

## 📈 Değerlendirme Metrikleri

- `Accuracy`
- `Precision, Recall, F1-Score`
- `ROC-AUC`
- `Confusion Matrix`

LightGBM modeli bu metriklerde en yüksek başarıyı sağlamıştır.

---

## 🏥 Gerçek Hayatta Kullanımı

- **Sağlık Kurumları**: Erken teşhis için destek sistemi
- **Sigorta Firmaları**: Risk değerlendirme modelleri
- **Mobil Uygulamalar**: Kişisel sağlık takibi ve öneri sistemleri
- **Araştırma**: Epidemiyolojik analizlerde veri sınıflandırma

---

## 🚀 Gelecekteki Geliştirme Fikirleri

- **Derin Öğrenme Uygulamaları**: MLP veya CNN tabanlı modeller
- **Gerçek Zamanlı Uygulama**: Giyilebilir cihazlardan gelen verilerle analiz
- **Yeni Veriler**: Farklı ülkelerden daha geniş veri ile modelin genellenmesi
- **Web Uygulaması**: Streamlit veya Flask ile kullanıcı arayüzü

---

## 🤝 Katkıda Bulunanlar

**Fatih Korkmaz**  
🔗 [Kaggle Profilim](https://www.kaggle.com/fatihkorkmaz7)

---


# MRG'de Bilgisayar Destekli Meme Tümörü Segmentasyonu İçin Geliştirilmiş Koşullu Çekişmeli Üretici Ağlar ve MONAI UNet

**Yazar:** Yasir Ömer Alparslan  
**Danışman:** Doç. Prof. Dr. Osman Özkaraca 
**Kurum:** Muğla Sıtkı Koçman Üniversitesi, Yapay Zeka Anabilim Dalı  

---

## 📌 Proje Özeti
Bu depo, DCE-MRI taramalarında yüksek derecede doğru ve kararlı meme tümörü segmentasyonu için MONAI UNet omurgası ile entegre edilmiş bir Koşullu Çekişmeli Üretici Ağın (cGAN) resmi uygulamasını içerir.

Bu proje, temel BTS-GAN mimarisini sistematik olarak hata ayıklama işleminden geçirir, iyileştirir ve yükseltir. Kararsız Paralel Dilate Konvolüsyon (PDC) bloklarını standartlaştırılmış, artık öğrenme (residual learning) tabanlı **MONAI UNet** ile değiştirerek, bu çerçeve gradyan kararsızlığını ve veri artırma desenkronizasyonu sorunlarını çözer ve üretime hazır bir Klinik Karar Destek Sistemi (KDDS) sunar.

---

## 🎯 Temel Özellikler ve Metodoloji
* **MONAI UNet Üreteci (Generator):** Geleneksel PDC modüllerini artık bağlantılarla (`num_res_units=2`) değiştirerek, gradyanların darboğaz katmanlarını atlamasına olanak tanır ve gradyan sönümlenmesi (vanishing gradient) problemini aktif olarak engeller.
* **PatchGAN Ayırt Edici (Discriminator):** Yüksek frekanslı yapısal tutarlılığı ve gerçekçi tümör sınırı belirlemeyi sağlar.
* **Senkronize Veri Artırma:** Görüntü-maske çiftlerinin hizalama hatalarını önlemek için eşzamanlı olarak artırılmasını sağlayan katı bir uzamsal dönüşüm hattı.
* **Hibrit Kayıp Topluluğu (Loss Ensemble):** Aşırı sınıf dengesizliğini yönetmek ve mekansal örtüşmeyi zorunlu kılmak için 5 bileşenli bir kayıp fonksiyonu aracılığıyla optimize edilmiştir:
  `Toplam Kayıp = BCE + 5*Dice + 2*Focal + 10*L1 + 0.2*GAN`

---

## 🗄️ Veri Kümesi Açıklaması (BreastDM)
Eğitim ve değerlendirme için kullanılan birincil veri kümesi **BreastDM Veri Kümesi**'dir (Meme Tümörü Görüntü Segmentasyonu ve Sınıflandırması için bir DCE-MRI Veri Kümesi).

### Veriler Nasıl Hazırlanır:
1. BreastDM veri kümesini indirin ve `SEG` alt kümesini bulun.
2. Orijinal görüntüleri ve bunlara karşılık gelen ikili maskeleri `data/raw/` dizinine yerleştirin.
3. Verileri formatlamak için ön işleme betiğini çalıştırın:
   * Görüntüler 512 x 512 piksel boyutuna yeniden ölçeklendirilir.
   * Maskeler kesin olarak {0, 1} uzayına eşlenir.
   * Kanal normalizasyonu uygulanır (μ=0.5, σ=0.5).
4. Veri kümesi otomatik olarak **%70 Eğitim**, **%15 Doğrulama** ve **%15 Test** olarak bölümlendirilir.

---

## 📂 Depo Yapısı (Repository Structure)

Proje, tekrarlanabilirliği ve temiz kod mimarisini sağlamak için aşağıdaki gibi düzenlenmiştir:

```text
mri-breast-tumor-cgan/
│
├── README.md                 # Proje özeti ve talimatlar
├── requirements.txt          # Python/PyTorch/MONAI bağımlılıkları
│
├── data/                     # Veri Kümesi Dizinleri (.gitignore ile hariç tutulmuştur)
│   ├── raw/                  # Orijinal BreastDM veri kümesi from this link https://drive.google.com/file/d/1GvNwL4iPcB2GRdK2n353bKiKV_Vnx7Qg/view
│   └── processed/            # 512x512 normalize edilmiş tensörler
│
├── src/                      # Kaynak Kodu
│   ├── dataset.py            # Veri Yükleyici (Dataloader) ve Senkronize Artırma 
│   ├── models/               # Ağ Mimarileri
│   │   ├── generator.py      # MONAI UNet entegrasyonu
│   │   └── discriminator.py  # PatchGAN mimarisi
│   ├── losses.py             # Hibrit Kayıp Topluluğu fonksiyonları
│   ├── train.py              # AMP destekli ana eğitim döngüsü
│   └── evaluate.py           # Test ve metrik hesaplamaları (DSC, IoU, vb.)
│
├── notebooks/                # Keşifsel Veri Analizi ve Görselleştirmeler
│   └── inference_visualization.ipynb # Kanser Katmanı ve sınır çizimleri
│
├── checkpoints/              # Kaydedilmiş model ağırlıkları (latest_checkpoint.pth)
│
└── reports/                  # Akademik raporlar ve oluşturulan grafikler
    ├── paper.tex             # Araştırma raporu için LaTeX kaynak kodu
    └── figures/              # Metrik grafikleri ve segmentasyon sonuçları
	
🚀 Kurulum ve Ön Koşullar
Bu projeyi çalıştırmak için CUDA destekli bir GPU'ya sahip bir sisteme ihtiyacınız vardır (NVIDIA RTX PRO 6000 Blackwell üzerinde geliştirilmiş ve test edilmiştir).

Depoyu klonlayın ve gerekli bağımlılıkları yükleyin:
# Depoyu klonlayın
git clone [https://github.com/yourusername/mri-breast-tumor-cgan.git](https://github.com/yourusername/mri-breast-tumor-cgan.git)
cd mri-breast-tumor-cgan

# Sanal bir ortam oluşturun (isteğe bağlı ancak önerilir)
python -m venv venv
source venv/bin/activate  # Windows'ta venv\Scripts\activate kullanın

# Bağımlılıkları yükleyin
pip install -r requirements.txt

Not: CUDA araç setiniz (toolkit) için PyTorch'un doğru sürümünün yüklü olduğundan emin olun.

💻 Kullanım
1. Veri Ön İşleme
Veri kümesi betiğini çalıştırarak verileri hazırlayın:
python src/dataset.py --input_dir data/raw --output_dir data/processed
2. Modeli Eğitme
Varsayılan hiperparametrelerle (Parti Boyutu = 8, Başlangıç Öğrenme Oranı = 1e-4) cGAN eğitimini başlatmak için:
python src/train.py --epochs 200 --batch_size 8 --early_stopping_patience 15
Eğitim betiği, Otomatik Karışık Hassasiyet (AMP) kullanır ve kesintiler durumunda checkpoints/latest_checkpoint.pth dosyasından otomatik durum kurtarmayı destekler.

3. Değerlendirme
Modeli daha önce görülmemiş %15'lik test seti üzerinde değerlendirmek için:
python src/evaluate.py --weights checkpoints/best_model.pth
4. Görselleştirme
Klinik "Kanser Katmanı (Cancer Overlay)" görselleştirmeleri oluşturmak için Jupyter Notebook'u kullanın:
jupyter notebook notebooks/inference_visualization.ipynb
📈 Sonuçlar ve PerformansModel, bağımsız bir test dağılımında (3.066 görüntü) titizlikle değerlendirilmiştir. Yanlış pozitif oranlarını önemli ölçüde azaltarak yüksek keskinlikli bir klinik profil sergiler.MetrikDeğerKlinik YorumDice (DSC)0.811Güçlü mekansal örtüşme, klinik eşiklerin oldukça üzerinde.IoU0.738Yüksek uzamsal kesişim doğruluğu.Keskinlik (Precision)0.886Dikkat çekici derecede düşük yanlış alarm oranı, klinisyen güvenini sağlar.Duyarlılık (Recall)0.810Gerçek pozitif tümör alanlarının %81'inin başarıyla izole edilmesi.📄 LisansBu proje akademik ve araştırma amaçlıdır. Verileri kullanırken lütfen BreastDM veri kümesi kullanım politikalarına uygunluğu sağlayın.🤝 TeşekkürBu araştırmayı mümkün kılmak için gerekli temel araçları ve verileri sağlayan Muğla Sıtkı Koçman Üniversitesi'ne, MONAI Konsorsiyumu'na ve BreastDM veri kümesini oluşturanlara özel teşekkürler.

note: after downloading data set from the link https://drive.google.com/file/d/1GvNwL4iPcB2GRdK2n353bKiKV_Vnx7Qg/view 
Data Format
The files should be putting as the following structure.

files
└── <dataset>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 001.png
        ├── 002.png
        ├── 003.png
        ├── ...

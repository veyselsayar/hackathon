ALGORİTMA KARMAŞIKLIĞI
1. Özellik Çıkarımı (extract_features)
Her görsel için sabit işlemler:

cv2.cvtColor ve cv2.resize: O(1)

Canny Edge: O(N) (N = toplam piksel = 64x64)

Histogram: O(N)

Simetri (ayna ile fark): O(N)

Flatten: O(N)

Hepsi birlikte O(N) (N: 4096)

Yani: Her görsel için lineer karmaşıklık

2. Döngüyle Veri Hazırlama
O(M x N)
(M = Görsel sayısı, N = 4096; yani her görsel için extract_features çağrısı)

3. Model Eğitimi (SVC)
Linear SVM için eğitim karmaşıklığı:

O(M x N x min(M, N))
(sklearn için pratikte; teorik olarak daha az, çünkü linear kernel seçtin)

M: Görsel sayısı (örnek), N: özellik sayısı

4. Prediction
Tek test için: O(N)

Tüm test seti için: O(T x N) (T = test seti uzunluğu)

5. Bellek (Mekân) Karmaşıklığı
Veri matrisi: O(M x N)

SVM Model parametreleri: O(N)

Ortalama RAM: Görüntü başına özellik vektörleri tutulduğu için, büyük veri setlerinde dikkat edilmeli.

Kısaca Sonuç
Zaman karmaşıklığı:
Özellik çıkarımı için O(M x N)
Linear SVM eğitimi için O(M x N x min(M, N))

Mekân karmaşıklığı:
O(M x N)

Kodun avantajı: Linear kernel seçildiği için yüksek boyutlu vektörlerde hızlı ve kararlı.

Ekstra: Özellik çıkarımı işlemi sabit olduğu için paralel programlama ile hızlandırılabilir.


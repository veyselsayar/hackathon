Adım  Zaman Karmaşıklığı	Mekân Karmaşıklığı
Özellik çıkarımı	O(N × M × N)	O(N × D)
SVM eğitimi	O(N × D × min(N, D))	O(N × D)
Tahmin (prediction)	O(D)	O(S × D)

N: toplam örnek/görsel sayısı

M × N: görsel boyutu

D: öznitelik boyutu (4130)

S: support vector sayısı


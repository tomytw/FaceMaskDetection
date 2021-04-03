Sumber Tutorial yang digunakan untuk membuat proyek ini:
- Youtube: https://www.youtube.com/watch?v=Ax6P93r32KU
- Github: https://github.com/balajisrinivas/Face-Mask-Detection


HOW TO USE (butuh tensorflow v2.3 karena menggunakan model yang di training di colab):
1. Buka file detect_mask.py
2. Ubah imageInputDir menjadi directory yang berisi input image-image yang ingin di deteksi menggunakan masker / tidak (secara default sudah tertulis folder input)
3. Jalankan detect_mask.py
4. Hasil akan muncul di layar lalu akan di save juga ke folder output

INFORMASI TAMBAHAN:
- Percobaan variasi training dilakukan menggunakan google collab (karena memakan waktu yang lama dan cpu cukup berat jika training di laptop)

- Contoh output image terdapat di folder output (menggunakan gambar-gambar yang berasal dari folder input), bounding box warna hijau berarti menggunakan masker, merah berarti tidak menggunakan masker

- Ubah variable useVideo menjadi True jika ingin menggunakan webcam untuk mendeteksi masker

- TERDAPAT BOUNDING BOX WARNA MERAH untuk orang yang TIDAK MENGGUNAKAN MASKER agar lebih jelas, jika bounding box untuk orang yang tidak memakai masker ingin dihilangkan, maka variable showFalseBox diubah ke False

- Untuk history training tiap epoch nya terdapat di file epoch.txt (selain itu variable history juga sudah di pickle jika ingin melihat statistik loss,val_loss,acc,val_acc di python)

- Untuk plot akurasi dan loss terdapat di file plot-new.png
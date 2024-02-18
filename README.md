# Predictive Analysis: Pulsar Prediction
By: Made Rahano Satryani Widhi

## Domain Proyek

Proyek ini berkaitan dengan klasifikasi *pulsar*, yaitu jenis bintang neutron langka yang menghasilkan emisi radio terdeteksi di Bumi. *pulsar* memiliki signifikansi ilmiah yang besar dalam penyelidikan ruang-waktu, medium antarbintang, dan keadaan materi. Dalam upaya untuk mempercepat analisis, alat machine learning digunakan untuk memberi label kandidat *pulsar* secara otomatis.

Klasifikasi pulsar dapat memberikan kontribusi signifikan pada penelitian ilmiah dalam ruang-waktu dan medium antarbintang melalui beberapa mekanisme. Berikut adalah beberapa cara di mana klasifikasi pulsar dapat memberikan wawasan yang berharga:

- Deteksi Gelombang Gravitasi:

  *Pulsar* dapat digunakan sebagai detektor gelombang gravitasi. Variasi waktu kedatangan *pulsar pulses* dapat menunjukkan adanya gelombang gravitasi yang melewati ruang-waktu. Dengan mengamati perubahan pola *pulse*, peneliti dapat memberikan kontribusi pada pemahaman tentang sifat dan sumber gelombang gravitasi [[1]](#1).

- Uji Teori Relativitas Umum:

  Pemeriksaan waktu kedatangan *pulsar pulses* memberikan kesempatan untuk menguji teori relativitas umum Albert Einstein. Variabilitas yang diamati dapat memberikan petunjuk tentang dampak gravitasi pada sifat-sifat pulsar, sehingga membantu mengonfirmasi atau menyesuaikan teori relativitas umum [[2]](#2).

- Penelitian Medium Antarbintang:

  *Pulsar* menghasilkan *pulse radio* yang dapat melewati medium antarbintang sebelum mencapai pengamat. Variabilitas dalam waktu kedatangan *pulse* dapat disebabkan oleh interaksi dengan medium ini, memberikan informasi tentang sifat dan distribusi medium antarbintang [[3]](#3).

## Business Understanding

### Problem Statements

- Keterbatasan Analisis Manual:

  Proses identifikasi *pulsar* secara manual memerlukan waktu dan upaya yang signifikan dari peneliti, menghambat kemajuan dalam penelitian ilmiah. Bagaimana caranya agar para peneliti dapat mengatasi keterbatasan analisis manual dalam proses identifikasi *pulsar*, yang memakan waktu dan upaya signifikan dari peneliti, sehingga dapat mengakselerasi kemajuan dalam penelitian ilmiah?

- Kompleksitas Karakteristik *pulsar*:

  Karakteristik *pulsar* dapat sangat kompleks, membuatnya sulit untuk dibedakan dari bintang lainnya hanya dengan metode analisis konvensional. Apakah ada metode analisis yang lebih efektif untuk mengatasi kompleksitas karakteristik *pulsar*, sehingga memudahkan pengenalan dan pembedaan dari bintang lain hanya dengan metode analisis konvensional?

- Kecepatan Proses Identifikasi:

  Bagaimana caranya agar dapat meningkatkan kecepatan proses identifikasi kandidat pulsar secara otomatis, khususnya dengan peningkatan volume data dari teleskop dan instrumen observasi, untuk menjawab tuntutan akan kecepatan dalam penelitian ilmiah?


### Goals

- Automatisasi Identifikasi *pulsar*:

  Mengembangkan sistem klasifikasi otomatis berbasis machine learning yang dapat secara akurat membedakan antara kandidat *pulsar* dan *non-pulsar*. Tujuan utama adalah mengurangi keterlibatan manusia dalam proses identifikasi dengan mencapai tingkat keandalan yang tinggi.

- Peningkatan Akurasi:

  Meningkatkan akurasi identifikasi *pulsar* dengan menerapkan dan mengoptimalkan algoritma pembelajaran mesin. Fokus khusus pada mengurangi tingkat false positives dan false negatives untuk memastikan bahwa hasil klasifikasi mencapai tingkat keakuratan yang tinggi dan dapat diandalkan.

- Efisiensi Analisis Data:

  Mengembangkan metode analisis data yang lebih efisien dengan mengimplementasikan teknik-teknik baru untuk mengidentifikasi dan memprioritaskan kandidat *pulsar* yang memiliki potensi tinggi. Tujuannya adalah meminimalkan waktu yang diperlukan oleh peneliti dalam proses identifikasi *pulsar*, sehingga mereka dapat lebih fokus pada analisis mendalam terhadap kandidat-kandidat yang lebih menjanjikan.

### Solution statements

- Menggunakan Algoritma Support Vector Machine (SVM)

  SVM adalah algoritma machine learning yang dikenal memiliki kemampuan klasifikasi yang tinggi, terutama untuk tugas klasifikasi biner. SVM dapat menangani dataset dengan dimensi tinggi dengan baik, yang sesuai dengan data kompleks seperti karakteristik *pulsar* yang diukur.

- Menggunakan Algoritma Support Vector Machine (SVM) dengan Kernel RBF

  - Penanganan Data Nonlinear: Kernel RBF memungkinkan SVM untuk menangani data yang tidak linear dan kompleks. Karakteristik *pulsar* mungkin memiliki pola yang tidak dapat dipisahkan dengan baik dengan menggunakan kernel linear.

  - Transformasi ke Dimensi Tak Terbatas: Kernel RBF mengubah data ke dimensi tak terbatas, memungkinkan SVM untuk menangkap pola yang lebih kompleks dan non-linear.

- Menggunakan Algoritma Support Vector Machine (SVM) dengan Kernel Linier

  - Kasus Linear Separable: Jika karakteristik *pulsar* dapat dipisahkan dengan baik menggunakan garis linear, kernel linier dapat memberikan hasil yang baik dengan komputasi yang lebih efisien.

  - Interpretasi Model yang Mudah: Kernel linier sering memberikan model yang lebih mudah diinterpretasikan, yang dapat menjadi keuntungan dalam pemahaman karakteristik *pulsar*.

## Data Understanding

Sumber Dataset : https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate/data

Setiap kandidat atau baris dijelaskan oleh 8 variabel kontinu, dan satu variabel kelas atau label. Empat yang pertama adalah statistik sederhana yang diperoleh dari *integrated pulse profile* (*folded profile*). Ini adalah susunan variabel kontinu yang mendeskripsikan versi sinyal dengan resolusi garis bujur yang telah dirata-ratakan dalam waktu dan frekuensi. Empat variabel sisanya diperoleh dengan cara yang sama dari kurva DM-SNR (*Dispersion Measure of the Signal to Noise Ratio*).

### Variabel-variabel pada *Pulsar Star* dataset adalah sebagai berikut:
- Mean of the integrated profile: nilai rata-rata dari *integrated pulse profile*, yang merupakan ringkasan dari sinyal radio yang diterima dari sebuah *pulsar* selama periode tertentu.
- Standard deviation of the integrated profile: mengukur seberapa besar variasi atau penyebaran data *integrated pulse profile*.
- Excess kurtosis of the integrated profile: Kurtosis mengukur "ekor" dari distribusi probabilitas. Kelebihan kurtosis membandingkan kurtosis distribusi dengan distribusi normal.
- Skewness of the integrated profile: Kemiringan mengukur asimetri dari distribusi probabilitas dari suatu variabel acak bernilai riil.
- Mean of the DM-SNR curve: rasio sinyal terhadap kebisingan rata-rata dari berbagai nilai dispersi dalam urutan *pulse* yang telah di-dedisperse.
- Standard deviation of the DM-SNR curve: mengukur variabilitas atau penyebaran dari rasio sinyal terhadap kebisingan di berbagai nilai dispersi.
- Excess kurtosis of the DM-SNR curve: sama seperti kelebihan kurtosis dari *integrated profile*, mengukur "ekor" dari distribusi probabilitas dari nilai DM-SNR.
- Skewness of the DM-SNR curve: mengukur asimetri dari distribusi probabilitas dari nilai DM-SNR.
- Class: Menentukan apakah data baris ini apakah bintang *pulsar* atau tidak.

### Exploratory Data Analysis

- Multivariate Analysis

  ![pulsar-correlation-matrix](https://github.com/maderahano/accom-image/assets/76169846/36bde737-e8a9-41b4-942c-c9a585a1e73d)
  Gambar 1. Correlation Matrix dari data *Pulsar*

  Pada gambar 1, korelasi matrix antara target dan delapan variabel menunjukkan pola hubungan yang dapat diinterpretasikan. Variabel IP_mean, IP_std, IP_kurtosis, IP_skewness, DM-SNR_mean, DM-SNR_std, DM-SNR_kurtosis, dan DM-SNR_skewness menunjukkan bahwa nilai korelasi dengna variabel target tidak ada yang mendekati 0. Mengingat bahwa pada gambar 1, baris yang paling akhir tidak ada warna yang mendekati putih, maka dapat disimpulkan bahwa semua 8 variabel tersebut berkolerasi dengan variabel target.

- Missing Value

  Untuk menangani nilai yang hilang dalam dataset, terdapat dua metode umum yang dapat digunakan, yaitu dropna() dan fillna(). Pemilihan antara keduanya bergantung pada situasi spesifik dalam dataset. Jika terdapat sejumlah besar nilai yang hilang, menggunakan dropna() mungkin bukan pilihan terbaik karena dapat menyebabkan kehilangan sejumlah besar data.

  ![pulsar-heatmap-missing-value](https://github.com/maderahano/accom-image/assets/76169846/fc854799-779c-435c-b7ba-27ba8e0a7e7d)
  Gambar 2. Heat Map Missing Value

  Jika dililhat pada gambar 2, terlihat bahwa distribusi nilai yang hilang tersebar merata, opsi yang baik adalah menggunakan metode ffill pada fillna(). Metode ffill, singkatan dari forward fill, bekerja dengan mengisi nilai yang hilang dengan nilai dari baris sebelumnya. Artinya, jika ada baris dengan nilai yang hilang, metode ffill akan mengambil nilai dari baris sebelumnya dan mengisinya ke dalam baris yang kosong tersebut.

- Outliers

  Dalam pendekatan untuk mendeteksi outlier, outlier diidentifikasi dengan menghitung batas bawah dan batas atas menggunakan kuartil pertama (Q1), kuartil ketiga (Q3), dan Interquartile Range (IQR). Nilai yang berada di luar batas ini dianggap sebagai outlier. Tujuan metode ini adalah untuk memberikan wawasan tentang distribusi data dan mencari nilai-nilai yang jauh dari kebanyakan data. Untuk hasilnya akan seperti pada gambar 3.

  ![pulsar-histplot-boxplot-outliers](https://github.com/maderahano/accom-image/assets/76169846/93e5e892-ba4c-4519-a38d-2572214f58e7) 
  Gambar 3. Histogram dan Boxplot dari data pulsar dengan outlier

  Dalam penanganan outlier, outlier ditangani dengan menggantikan nilai-nilai yang berada di luar batas bawah dan batas atas dengan nilai batas tersebut. Pendekatan ini dilakukan untuk meminimalkan pengaruh outlier terhadap analisis statistik dan visualisasi data. Dengan menggantikan nilai outlier, distribusi data menjadi lebih representatif, sehingga analisis lebih stabil dan akurat. Setelah diterapkan, hasilnya akan jadi seperti pada gambar 4.

  ![pulsar-histplot-boxplot-without-outliers](https://github.com/maderahano/accom-image/assets/76169846/bdd491ad-ab9c-4c1a-847d-ccb06bbeadb6)
  Gambar 4. Histogram dan Boxplot dari data pulsar tanpa outlier

## Data Preparation

- Train-Test Split

  Proses train-test split merupakan langkah untuk membagi dataset menjadi dua subset utama: data latih (train set) dan data uji (test set). Proporsi pembagian menjadi data latih dan data uji memiliki dampak signifikan terhadap kualitas dan keandalan model yang dihasilkan. Dalam konteks proyek ini, dipilih proporsi 90% data latih dan 10% data validasi. Keputusan ini didasarkan pada pertimbangan untuk memberikan model informasi yang memadai untuk pelatihan, mengurangi risiko overfitting, dan memungkinkan validasi yang objektif terhadap performa model pada data yang tidak terlibat dalam pelatihan. Proporsi ini juga memastikan bahwa model dapat menggeneralisasi dengan baik pada data baru. Alternatif proporsi, seperti 80% data latih dan 20% data uji, serta metode validasi silang (cross-validation), juga dipertimbangkan sebagai opsi yang dapat memengaruhi evaluasi dan kinerja model. Penggunaan proporsi pembagian yang tepat adalah kunci untuk menghasilkan model yang akurat dan dapat diandalkan.

- Standarisasi

  Proses standarisasi melibatkan transformasi distribusi nilai pada setiap fitur atau variabel dalam dataset sehingga nilai rata-rata menjadi 0 dan deviasi standar menjadi 1. Tindakan ini diperlukan untuk memastikan keseragaman skala pada setiap data numerik. Menyamakan skala ini penting agar model tidak dipengaruhi secara tidak proporsional oleh fitur-fitur yang memiliki skala yang lebih besar. Hal ini dapat menghindari situasi di mana beberapa fitur mendominasi yang lainnya hanya karena perbedaan skala. Selain itu, standarisasi juga dapat meningkatkan kinerja model dengan memudahkan algoritma untuk menginterpretasikan dan mengambil keputusan yang lebih baik terhadap data yang telah disesuaikan dengan skala yang seragam. Dengan menggunakan standarisasi, proyek ini dapat memastikan bahwa setiap fitur memberikan kontribusi yang seimbang terhadap pembentukan model, meminimalkan potensi bias yang mungkin muncul akibat perbedaan skala.

## Modeling
Pada tahap modeling proyek ini, algoritma utama yang digunakan adalah Support Vector Machine (SVM), yang memiliki kemampuan klasifikasi tinggi dan dapat menangani dataset dengan dimensi tinggi. Dalam eksperimen ini, dua jenis kernel SVM dieksplorasi, yaitu Radial Basis Function (RBF) dan Linear, untuk memahami sekaligus sifat linear dan non-linear dari karakteristik pulsar.

- Tahapan dan Parameter:

  Dalam penggunaan SVM dengan kernel RBF dan Linier, proses modeling dilakukan dengan mempertimbangkan beberapa parameter kunci. Pada kernel RBF, parameter gamma dan C menjadi fokus utama untuk mengatur kompleksitas model dan menangani overfitting. Sedangkan pada kernel Linier, parameter C yang mengendalikan trade-off antara margin dan kesalahan klasifikasi menjadi perhatian utama. Jika diuraikan secara singkat, maka nilai parameter yang digunakan dalam eksperimen ini adalah sebagai berikut:
  - Untuk Kernel RBF:
    - C (Cost): Menentukan trade-off antara kesalahan klasifikasi dan kompleksitas model. Nilai yang diuji dalam eksperimen ini adalah [0.01, 0.1, 1, 10].
    - Gamma: Mengatur bentuk kurva keputusan. Nilai yang diuji dalam eksperimen ini adalah [0.09, 0.1, 0.2, 0.001].
  - Untuk Kernel Linear:
    - C (Cost): Menentukan trade-off antara kesalahan klasifikasi dan kompleksitas model. Nilai yang diuji dalam eksperimen ini adalah [0.01, 0.1, 1, 10].

  Semua paramter tersebut akan dicari paramter yang terbaik dengan menggunakan metode GridSearchCV. Hasil dari proses GridSearch untuk Support Vector Machine (SVM) dengan kernel RBF maupun Linear menunjukkan bahwa parameter terbaik untuk model adalah sebagai berikut:

  - C (Cost): 10
  - Gamma: 0.1

  Dengan parameter ini, model SVM mampu memberikan klasifikasi yang optimal dan efisien berdasarkan data yang diberikan. Nilai C yang tinggi (10) menunjukkan penekanan pada margin yang lebih ketat dan mengurangi toleransi terhadap kesalahan klasifikasi. Gamma sebesar 0.1 mengindikasikan bentuk kurva keputusan yang optimal untuk data ini.

- Kelebihan dan Kekurangan:

  Kernel RBF memiliki kelebihan efektivitas dalam menangani data non-linear dan kompleks. Namun, kelemahannya melibatkan sensitivitas terhadap pemilihan parameter yang memerlukan tuning yang cermat. Di sisi lain, kernel Linear memberikan model yang lebih mudah diinterpretasikan, terutama pada data yang dapat dipisahkan secara linear. Namun, kekurangannya mungkin terletak pada keterbatasannya dalam menangani pola non-linear pada data.

- Situasi Ketika Satu Kernel Lebih Cocok:

  Dalam beberapa situasi, satu kernel mungkin lebih cocok daripada yang lain tergantung pada karakteristik data. Misalnya, jika data memiliki pola yang sangat kompleks dan non-linear, kernel RBF mungkin lebih unggul, sementara pada data dengan struktur yang lebih sederhana, kernel Linear bisa menjadi pilihan lebih baik.

- Proses Improvement:

  Dalam upaya untuk meningkatkan kinerja model, dilakukan proses improvement dengan menggunakan metode Grid Search. Metode ini memungkinkan penelitian sistematis pada berbagai kombinasi parameter, seperti C dan gamma pada kernel RBF, serta C pada kernel Linier. Dengan melakukan pencarian parameter secara otomatis, proyek ini dapat memilih parameter terbaik yang menghasilkan model SVM yang optimal dan efisien.

## Evaluation
Dalam melakukan evaluasi performa model klasifikasi pulsar, dengan menggunakan beberapa metrik evaluasi yang sesuai dengan konteks data dan problem statement. Metrik yang dipilih melibatkan precision, recall, F1-score, dan support.

### Penjelasan mengenai metrik yang digunakan

- Precision (Presisi): Menunjukkan sejauh mana hasil positif yang diprediksi oleh model benar-benar positif. Formulanya adalah precision = TP / (TP + FP), di mana TP adalah True Positive dan FP adalah False Positive.

- Recall (Recall atau Sensitivitas): Menunjukkan sejauh mana model dapat menemukan kembali semua instance yang positif. Formulanya adalah recall = TP / (TP + FN), di mana TP adalah True Positive dan FN adalah False Negative.

- F1-score (Skor F1): Merupakan harmonic mean dari precision dan recall, memberikan gambaran seimbang antara keduanya. Formulanya adalah F1-score = 2 * (precision * recall) / (precision + recall).

- Support: Jumlah instance dalam kelas target.

### Hasil proyek berdasarkan metrik evaluasi

#### SVM dengan RBF Kernel

- Hasil klasifikasi metriks dari training data:

  Tabel 1. Evaluasi training data pulsar dengan RBF Kernel
  |               | Precision | Recall | f1-score | support |
  | ------------- | --------- | ------ | -------- | ------- |
  | 0             | 0.98      | 0.99   | 0.99     | 10242   |
  | 1             | 0.94      | 0.84   | 0.89     | 1033    |
  | accuracy      | -         | -      | 0.98     | 11275   |
  | macro avg     | 0.96      | 0.92   | 0.94     | 11275   |
  | weighted avg  | 0.98      | 0.98   | 0.98     | 11275   |

  Evaluasi pada training data menunjukkan kinerja model yang kuat. Dengan precision sebesar 0.98 untuk kelas non-pulsar dan 0.94 untuk kelas pulsar, model mampu memberikan prediksi yang akurat. Tingginya recall pada kelas non-pulsar (0.99) menandakan kemampuan model mengidentifikasi non-pulsar dengan sangat baik, sementara recall pada kelas pulsar (0.84) menunjukkan model memiliki potensi untuk lebih memperbaiki kemampuannya dalam mengenali pulsar. Akurasi keseluruhan sebesar 0.98 memberikan gambaran positif tentang keandalan model pada data training. F1-score yang seimbang pada kelas non-pulsar (0.99) menunjukkan kualitas prediksi yang optimal, meskipun terdapat ruang untuk peningkatan pada kelas pulsar (0.89). Dengan nilai macro average sebesar 0.94 dan weighted average sebesar 0.98 untuk precision, recall, dan F1-score, dapat disimpulkan bahwa model SVM dengan kernel RBF berhasil mengklasifikasikan data training dengan baik.

- Hasil klasifikasi metriks dari testing data:

  Tabel 2. Evaluasi testing data pulsar dengan RBF Kernel
  |               | Precision | Recall | f1-score | support |
  | ------------- | --------- | ------ | -------- | ------- |
  | 0             | 0.98      | 0.99   | 0.99     | 1133    |
  | 1             | 0.91      | 0.80   | 0.85     | 120     |
  | accuracy      | -         | -      | 0.97     | 1253    |
  | macro avg     | 0.94      | 0.90   | 0.92     | 1253    |
  | weighted avg  | 0.97      | 0.97   | 0.97     | 1253    |

  Evaluasi pada testing data juga memberikan hasil yang bagus. Dengan precision sebesar 0.98 untuk kelas non-pulsar dan 0.91 untuk kelas pulsar, model mampu mempertahankan tingkat akurasi yang tinggi. Meskipun recall pada kelas pulsar (0.80) menurun dibandingkan dengan training data, akurasi keseluruhan masih mencapai 0.97. F1-score yang baik pada kelas non-pulsar (0.99) menunjukkan kemampuan model untuk memberikan prediksi yang konsisten, meskipun terdapat ruang untuk perbaikan pada kelas pulsar (0.85). Dengan nilai macro average sebesar 0.92 dan weighted average sebesar 0.97 untuk precision, recall, dan F1-score, model RBF Kernel secara umum berhasil mempertahankan kinerja yang baik pada data testing, menunjukkan kemampuan generalisasi yang baik dari model ini.

#### SVM dengan Linier Kernel

- Hasil klasifikasi metriks dari training data:

  Tabel 3. Evaluasi training data pulsar dengan Linier Kernel
  |               | Precision | Recall | f1-score | support |
  | ------------- | --------- | ------ | -------- | ------- |
  | 0             | 0.98      | 0.99   | 0.99     | 10242   |
  | 1             | 0.90      | 0.81   | 0.85     | 1033    |
  | accuracy      | -         | -      | 0.97     | 11275   |
  | macro avg     | 0.94      | 0.90   | 0.92     | 11275   |
  | weighted avg  | 0.97      | 0.97   | 0.97     | 11275   |

  Evaluasi pada training data menunjukkan performa model yang konsisten. Dengan precision sebesar 0.98 untuk kelas non-pulsar dan 0.90 untuk kelas pulsar, model memberikan prediksi yang akurat pada kedua kelas. Tingginya recall pada kelas non-pulsar (0.99) menandakan kemampuan model mengidentifikasi non-pulsar dengan sangat baik, sementara recall pada kelas pulsar (0.81) menunjukkan model dapat dengan baik mengenali pulsar, meskipun dengan tingkat keakuratan yang sedikit lebih rendah. Akurasi keseluruhan sebesar 0.97 memberikan gambaran positif tentang keandalan model pada data training. F1-score yang seimbang pada kelas non-pulsar (0.99) menunjukkan kualitas prediksi yang optimal, sementara F1-score pada kelas pulsar (0.85) menunjukkan keseimbangan yang baik antara precision dan recall.

- Hasil klasifikasi metriks dari testing data:

  Tabel 4. Evaluasi testing data pulsar dengan Linier Kernel
  |               | Precision | Recall | f1-score | support |
  | ------------- | --------- | ------ | -------- | ------- |
  | 0             | 0.97      | 0.99   | 0.98     | 1133    |
  | 1             | 0.90      | 0.76   | 0.82     | 120     |
  | accuracy      | -         | -      | 0.97     | 1253    |
  | macro avg     | 0.94      | 0.87   | 0.90     | 1253    |
  | weighted avg  | 0.97      | 0.97   | 0.97     | 1253    |

  Evaluasi pada testing data juga memberikan hasil yang baik. Dengan precision sebesar 0.97 untuk kelas non-pulsar dan 0.90 untuk kelas pulsar, model Linear Kernel mampu mempertahankan tingkat akurasi yang tinggi. Recall pada kelas pulsar (0.76) menurun sedikit dibandingkan dengan training data, namun akurasi keseluruhan tetap tinggi pada nilai 0.97. F1-score yang baik pada kelas non-pulsar (0.98) menunjukkan kemampuan model untuk memberikan prediksi yang konsisten, meskipun terdapat ruang untuk perbaikan pada kelas pulsar (0.82). Dengan nilai macro average sebesar 0.90 dan weighted average sebesar 0.97 untuk precision, recall, dan F1-score, model Linear Kernel secara umum mempertahankan kinerja yang baik pada data testing, menunjukkan kemampuan generalisasi yang memadai dari model ini.

### Penjelasan formula metrik dan bagaimana metrik tersebut bekerja.

- Precision (Presisi): Dalam konteks ini, precision penting untuk memastikan bahwa ketika model mengklasifikasikan kandidat sebagai pulsar, itu sebenarnya pulsar dan bukan false positive. Precision berguna dalam mengidentifikasi sejauh mana model dapat memberikan prediksi positif yang benar.

- Recall (Recall atau Sensitivitas): Recall penting untuk memastikan bahwa model dapat menemukan kembali sebagian besar kandidat pulsar yang sebenarnya ada dalam dataset. Recall berguna untuk mengidentifikasi sejauh mana model dapat menangkap pulsar yang sebenarnya, mengurangi false negatives.

- F1-score (Skor F1): F1-score memberikan gambaran seimbang antara precision dan recall. Ini adalah metrik yang baik untuk menilai performa model secara keseluruhan, terutama ketika ingin menghindari terlalu banyak false positives dan false negatives.

- Support: Jumlah support memberikan konteks tentang seberapa umum kelas target dalam dataset. Ini penting untuk memahami distribusi kelas dan seberapa representatif model terhadap kelas tertentu.

## Conclusion

Berdasarkan hasil matriks evaluasi dari kedua kernel, dapat diambil beberapa kesimpulan tentang performa model pada identifikasi pulsar menggunakan Support Vector Machine (SVM).

Pertama, kernel RBF menunjukkan kinerja yang sangat baik pada data training, dengan precision dan recall yang tinggi untuk kelas non-pulsar, serta akurasi keseluruhan sebesar 0.98. Meskipun kelas pulsar memiliki recall yang sedikit lebih rendah, model tetap memberikan hasil yang solid. Hasil yang serupa juga dapat diamati pada data testing, dengan model RBF Kernel mempertahankan tingkat akurasi yang tinggi. Keseluruhan, kernel RBF mampu mengklasifikasikan data pulsar dengan baik, terutama pada kelas non-pulsar.

Di sisi lain, kernel Linear menunjukkan kinerja yang konsisten pada kedua dataset, baik pada training maupun testing. Dengan akurasi keseluruhan sebesar 0.97, model Linear Kernel mampu memberikan prediksi yang konsisten dan andal. Meskipun recall kelas pulsar sedikit lebih rendah dibandingkan kernel RBF, model Linear Kernel masih memberikan hasil yang baik dan menunjukkan kemampuan generalisasi yang memadai pada data testing.

Kesimpulannya, kedua kernel SVM, baik RBF maupun Linear, memberikan hasil yang memuaskan dalam mengidentifikasi pulsar. Pilihan antara keduanya mungkin tergantung pada karakteristik data dan kebutuhan spesifik proyek. Kernel RBF cenderung lebih unggul dalam menangani data non-linear, sementara kernel Linear memberikan model yang lebih mudah diinterpretasikan. Dengan pemahaman ini, pemilihan kernel dapat disesuaikan dengan tujuan analisis dan karakteristik dataset yang dihadapi.

## Bibliography

<a id="1">[1]</a>  Das, Arpan, et al. “Pulsars as Weber Gravitational Wave Detectors.” Physics Letters B, vol. 791, Apr. 2019, pp. 167–171, https://doi.org/10.1016/j.physletb.2019.02.031. Accessed 23 Oct. 2022.

<a id="2">[2]</a> Stairs, Ingrid H. “Testing General Relativity with Pulsar Timing.” Living Reviews in Relativity, vol. 6, no. 1, 9 Sept. 2003, https://doi.org/10.12942/lrr-2003-5. Accessed 28 Nov. 2019.

<a id="3">[3]</a> Marschke, Laura. “Pulsars: A Key to Unlocking the Interstellar Medium.” 2006.
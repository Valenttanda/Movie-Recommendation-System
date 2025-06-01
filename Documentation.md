# Laporan Proyek Machine Learning - Mohammad Valeriant Qumara Tanda

## Project Overview

Dalam era digital yang semakin berkembang, sistem rekomendasi telah menjadi komponen penting dalam berbagai platform, termasuk layanan streaming film seperti Netflix, Disney+, dan Amazon Prime. Sistem ini berperan besar dalam membantu pengguna menemukan konten yang sesuai dengan preferensi mereka tanpa harus mencarinya secara manual. Salah satu pendekatan yang umum digunakan dalam sistem rekomendasi adalah _content-based filtering_, yaitu metode yang memberikan rekomendasi berdasarkan kesamaan konten antar item.

Proyek ini bertujuan untuk membangun sistem rekomendasi film berbasis konten dengan pendekatan content-based filtering, menggunakan dua model utama: TF-IDF (_Term Frequency–Inverse Document Frequency_) dan BERT (_Bidirectional Encoder Representations from Transformers_). Kedua model ini diterapkan untuk mengolah dua jenis fitur yang berbeda, yaitu `overview` (deskripsi film) dan `genres` (genre film). Dengan memanfaatkan _cosine similarity_, sistem diharapkan dapat mengukur kedekatan antar film dan memberikan rekomendasi yang relevan terhadap satu film masukan, yakni **Avatar**.

Alasan proyek ini penting adalah karena sistem rekomendasi yang baik dapat meningkatkan keterlibatan pengguna, mengurangi beban pencarian, dan memberikan pengalaman personalisasi yang lebih baik. Selain itu, dengan membandingkan dua pendekatan berbeda (statistik klasik dan representasi semantik dari _deep learning_), proyek ini dapat memberikan wawasan yang berharga tentang keefektifan masing-masing metode dalam konteks sistem rekomendasi film.

Sebagai referensi, pendekatan content-based filtering telah banyak dibahas dalam literatur akademik, seperti pada karya Lops et al. [1], yang membahas teknik dan tantangan dalam implementasi sistem rekomendasi berbasis konten. Di sisi lain, pemanfaatan BERT dalam sistem rekomendasi merupakan pendekatan yang relatif baru, namun menjanjikan dalam memahami makna semantik secara lebih mendalam, sebagaimana dijelaskan dalam penelitian Zhang et al. [2].

## Business Understanding

### Problem Statements

1. Bagaimana cara memberikan rekomendasi film yang relevan kepada pengguna berdasarkan film yang mereka sukai?
2. Model content-based seperti apa yang paling efektif dalam memberikan rekomendasi film berdasarkan deskripsi dan genre film?
3. Bagaimana bentuk evaluasi dari model sistem rekomendasi yang dibangun?

### Goals

1. Membangun sistem rekomendasi film berbasis konten yang mampu memberikan rekomendasi berdasarkan satu film masukan, menggunakan fitur `overview` dan `genre`.
2. Membandingkan performa dua pendekatan model *content-based*, yaitu TF-IDF (pendekatan statistik) dan BERT (pendekatan berbasis representasi semantik).
3. Melakukan evaluasi dengan metrik *Mean Reciprocal Rank* (MRR) dan *Normalized Discounted Cumulative Gain* (NDCG), serta evaluasi manual (melihat langsung pola film dari fitur `overview` dan `genres`) untuk menilai efektifitas sistem rekomendasi.

### Solution Statements

Untuk menjawab permasalahan di atas, solusi yang diterapkan dalam proyek ini meliputi:

- Pendekatan 1: TF-IDF + Cosine Similarity
  Pendekatan klasik ini memanfaatkan frekuensi kata dalam deskripsi film untuk menghitung kemiripan antar film. Cocok untuk fitur berbasis teks seperti `overview`.

- Pendekatan 2: BERT + Cosine Similarity
  Menggunakan *pre-trained transformer* (BERT) untuk menghasilkan representasi semantik dari teks. Representasi ini lebih peka terhadap konteks dan makna kalimat, sehingga diharapkan dapat memberikan rekomendasi yang lebih akurat dan relevan.

Kedua pendekatan digunakan pada dua fitur berbeda:
`overview`: deskripsi film
`genres`: kategori genre film

Perbandingan hasil dari keempat skema tersebut (TF-IDF overview, TF-IDF genres, BERT overview, dan BERT genres) menjadi dasar penentuan efektivitas pendekatan.

## Data Understanding

Proyek ini menggunakan dataset film dari [TMDB (The Movie Database)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), yang merupakan salah satu sumber data film terbuka dan populer. Dataset ini mencakup berbagai informasi tentang ribuan film, seperti judul, sinopsis (*overview*), genre, pendapatan, tanggal rilis, dan sebagainya. Dataset ini terdiri dari dua file utama: `tmdb_5000_movies.csv` dan `tmdb_5000_credits.csv`. Namun, untuk proyek ini, dataset yang digunakan hanya `tmdb_5000_movies.csv`, karena semua data dalam `tmdb_5000_credits.csv` tidak dibutuhkan dalam sistem rekomendasi ini.

### Informasi Dataset

Berikut adalah informasi dataset yang digunakan:

- Nama Dataset: `tmdb_5000_movies.csv`
- Sumber Dataset: [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Jumlah Dataset: 4803 Baris x 20 Kolom
- Kondisi Dataset:
  - Terdapat beberapa nilai kosong pada kolom seperti `overview`, `tagline`, dan `homepage`.
  - Kolom `genres`, `keywords`, `cast`, dan `crew` awalnya dalam format string JSON dan telah dikonversi menjadi list Python.
  - Semua teks dari fitur teks (`overview`, `genres`, dan lainnya) telah diubah menjadi huruf kecil (lowercase) dan dibersihkan dari karakter non-alfabet untuk memudahkan proses ekstraksi fitur teks.
  
### Statistik Umum

Beberapa statistik dasar yang diperoleh dari dataset yaitu:

- Rata-rata durasi film: 106.87 menit, dengan durasi terpanjang: 338 menit.
- Rata-rata skor pengguna (vote average): 6.09 dari maksimum 10.
- Film dengan pendapatan tertinggi: lebih dari $2.7 miliar.

### Variabel dalam Dataset

Berikut adalah fitur yang ada pada dataset `tmdb_5000_movies.csv`:

   - `budget`: Biaya produksi film.
   - `genres`: Daftar genre film dalam format string JSON.
   - `homepage`: Alamat situs web resmi film.
   - `keywords`: Daftar kata kunci film dalam format string JSON.
   - `id`: ID unik untuk setiap film.
   - `original_language`: Bahasa asli yang digunakan dalam film.
   - `original_title`: Judul asli film.
   - `overview`: Deskripsi singkat atau sinopsis film.
   - `popularity`: Tingkat kepopuleran suatu film.
   - `production_companies`: Daftar perusahaan produksi film dalam format string JSON.
   - `production_countries`: Daftar negara produksi film dalam format string JSON.
   - `release_date`: Tanggal rilis film.
   - `revenue`: Pendapatan film.
   - `runtime`: Durasi film dalam menit.
   - `spoken_language`: Bahasa yang digunakan dalam film.
   - `status`: Status film (produksi, rilis, dsb.).
   - `tagline`: Slogan film.
   - `title`: Judul film.
   - `vote_average`: Skor rata-rata berdasarkan suara pengguna.
   - `vote_count`: Jumlah suara yang diberikan pengguna.

## Data Preparation

Pada tahap ini, dilakukan sejumlah proses persiapan data untuk memastikan fitur-fitur yang digunakan dalam sistem rekomendasi dapat diolah secara optimal. Tahapan difokuskan pada pengolahan dataset `tmdb_5000_movies.csv`.

1. Seleksi Fitur Relevan

  Hanya fitur-fitur tertentu yang digunakan untuk membangun sistem rekomendasi, yaitu:

- `original_title`: Judul asli film, digunakan untuk keperluan tampilan dan pencocokan input.
- `overview`: Sinopsis film, digunakan sebagai dasar fitur konten berbasis teks.
- `genres`: Genre film, digunakan untuk fitur tambahan dalam rekomendasi.

2. Pembersihan dan Pra-Pemrosesan Teks

  Beberapa tahapan dilakukan terhadap data mentah agar bisa digunakan oleh model:
  a. Overview
     - Nilai kosong diisi dengan string kosong (`''`).
     - Teks diubah menjadi huruf kecil (lowercase).
     - Karakter non-alfabet dihapus untuk menghindari gangguan dalam proses vektorisasi teks.
  b. Genres
     - Format JSON string diubah menjadi list nama genre.
       - Contoh: '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]' menjadi ['action', 'dventure']
     - Genre digabungkan menjadi satu string (misalnya: "action adventure") agar dapat diproses sebagai dokumen teks.
  
```python
def clean_genres(genre_str):
  try:
    genres = ast.literal_eval(genre_str)
    return ' '.join([g['name'] for g in genres]).lower()
  except:
    return ''
df['genres'] = df['genres'].apply(clean_genres)
```

1. Penyesuaian Format untuk Model Teks
   Fitur `overview` dan `genres` kemudian diproses menjadi bentuk vektor menggunakan dua pendekatan yang berbeda:
   - TF-IDF: Menghitung frekuensi term untuk membuat representasi teks berbasis statistik.
   - BERT: Menghasilkan embedding semantik menggunakan model pretrained `sentence-transformers`.
   Hasil dari proses ini adalah matriks vektor yang siap digunakan untuk menghitung cosine similarity antar film.

## Modeling

Pada tahap ini, dibangun sistem rekomendasi berbasis konten (*content-based filtering*) yang bertujuan untuk memberikan rekomendasi film berdasarkan kemiripan deskripsi (`overview`) dan genre (`genres`) terhadap sebuah film masukan, yaitu **Avatar**.

Dua pendekatan utama digunakan dalam proyek ini:

1. Pendekatan 1: TF-IDF + Cosine Similarity
  a. Deskripsi
  Pendekatan ini menggunakan teknik *Term Frequency–Inverse Document Frequency* (TF-IDF) untuk mengubah teks `overview` dan `genres` menjadi representasi numerik. Kemudian digunakan cosine similarity untuk menghitung kemiripan antar film.
  b. Implementasi
  - TF-IDF pada Overview: Sinopsis film diubah menjadi vektor kata berbasis frekuensi, lalu dihitung kemiripannya menggunakan cosine similarity.
  - TF-IDF pada Genres: Genre film dikonversi menjadi string dan diperlakukan seperti dokumen teks, kemudian diproses dengan TF-IDF.

```python
tfv_overview = TfidfVectorizer()
tfidf_overview_matrix = tfv_overview.fit_transform(df['overview'])
cos_tfidf_overview = cosine_similarity(tfidf_overview_matrix)
indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()

def recommend_tfidf_cossim_overview(title, cos_tfidf_overview=cos_tfidf_overview):
  # Get the original_title index
  idx = indices[title]
  
  # Get the pairwise similarity scores
  cos_tfidf_overview_scores = list(enumerate(cos_tfidf_overview[idx]))
  
  # Sort movies
  cos_tfidf_overview_scores = sorted(cos_tfidf_overview_scores, key=lambda x: x[1], reverse=True)
  
  # Scores of the top 10 most similar movies
  cos_tfidf_overview_scores = cos_tfidf_overview_scores[1:6]
  
  # Movie indices 
  movie_indices = [i[0] for i in cos_tfidf_overview_scores]
  
  return df['original_title'].iloc[movie_indices]
```

  c. Output
  Model menghasilkan daftar film yang paling mirip dengan *Avatar*, baik berdasarkan `overview` maupun `genres`.
  Hasil rekomendasi (overview - TF-IDF) untuk 5 film teratas:

  - Apollo 18
  - Tears of the Sun
  - The American
  - Obitaemyy Ostrov
  - The Matrix

  [Hasil Rekomendasi TF-IDF overview](img/Screenshot 2025-06-01 092136.png)
  Nilai cosine similarity untuk kelima hasil:
  [Cosine Similarity Kelima Hasil](img/Screenshot 2025-06-01 092006.png)  

  Hasil rekomendasi (genres - TF-IDF) untuk 5 film teratas:

  - Superman Returns
  - Man of Steel
  - X-Men: Days of Future Past
  - Jupiter Ascending
  - The Wolverine

  [Hasil Rekomendasi TF-IDF genres](img/Screenshot 2025-06-01 092129.png)
  Nilai cosine similarity untuk kelima hasil:
  [Cosine Similarity Kelima Hasil](img/Screenshot 2025-06-01 091945.png)

2. Pendekatan 2: BERT (SBERT) + Cosine Similarity
   a. Deskripsi
   Menggunakan model pretrained BERT dari library `sentence-transformers`, pendekatan ini menghasilkan representasi semantik dari teks yang lebih kontekstual dibandingkan TF-IDF. Hal ini memungkinkan model memahami makna kalimat, bukan sekadar frekuensi kata.
   b. Implementasi
   - BERT pada Overview: Setiap sinopsis diproses menggunakan `sentence-transformers` (model `all-MiniLM-L6-v2`) untuk menghasilkan embeddings. Kemudian dihitung cosine similarity antar embedding.
   - BERT pada Genres: Meskipun `genres` adalah fitur pendek dan tidak berkalimat, pendekatan ini tetap digunakan untuk melihat perbedaan hasil.
  
```python
model_genres = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_genres = model_genres.encode(df['genres'].tolist(), show_progress_bar=True)
cos_bert_genres = cosine_similarity(embeddings_genres, embeddings_genres)

def recommend_bert_genres(title, top_n):
  idx = df[df['original_title'].str.lower() == title.lower()].index[0]
  # Pick recommended rows
  row = cos_bert_genres[idx]
   
  # Sort by the most similar
  similar_indices = row.flatten().argsort()[::-1]
  
  # Drop it's own movie
  similar_indices = similar_indices[similar_indices != idx]
    
  # Pick top_n
  top_indices = similar_indices[:top_n]
    
  # Take a result
  results = []
  for i in top_indices:
    results.append((df.iloc[i]['original_title'], round(row[i], 4)))
  return results
```

   c. Output
   Rekomendasi berdasarkan hasil kemiripan embedding BERT terhadap sinopsis dan genre dari *Avatar*.
   Hasil rekomendasi (overview - BERT) untuk 5 film teratas:

   - Alien: Resurrection
   - The Black Hole
   - Serenity
   - Aliens
   - Supernova

   [Hasil Rekomendasi BERT overview](img/Screenshot 2025-06-01 092102.png)

   Hasil rekomendasi (genres - BERT) untuk 5 film teratas:

   - X-Men: Days of Future Past
   - Man of Steel
   - Superman
   - Beastmaster 2: Through the Portal of Time
   - Superman II

   [Hasil Rekomendasi BERT genres](img/Screenshot 2025-06-01 092112.png)

3. Kelebihan dan Kekurangan Kedua Pendekatan

| Aspek | TF-IDF + Cosine Similarity | BERT + Cosine Similarity |
| ----- | -------------------------- | ------------------------ |
| **Kelebihan** | | |
| | Cepat dan efisien untuk dataset kecil-menengah  | Memahami konteks dan makna teks |
| | Implementasi sederhana dan tidak membutuhkan training | Mampu menangkap kemiripan semantik |
| | Cocok untuk teks eksplisit seperti sinopsis dan genre | Hasil lebih relevan secara tematik (terutama untuk `overview`) |
| **Kekurangan** | | |
|    | - Tidak memahami konteks atau makna kata | - Proses lebih lambat karena kompleksitas model |
| | Tidak mengenali sinonim | Tidak optimal untuk input pendek |
| | Kurang efektif untuk teks pendek seperti `genres` | Bergantung pada model pretrained BERT |

## Evaluation

Evaluasi dilakukan untuk mengukur sejauh mana sistem rekomendasi dapat memberikan hasil yang relevan terhadap satu input film, yaitu Avatar. Dua pendekatan utama digunakan dalam proses evaluasi ini:

- Evaluasi kuantitatif menggunakan metrik ranking (MRR dan nDCG)
- Evaluasi kualitatif (manual) berdasarkan ground truth dan inspeksi hasil rekomendasi

1. Pembuatan Ground Truth
Karena tidak tersedia data interaksi pengguna atau label relevansi eksplisit, maka ground truth disusun secara manual berdasarkan tingkat kepopuleran film dan tetap mempertahankan fitur `overview` dan `genres`.

```python
# Parsing 'genres' column to list
df['genres'] = df['genres'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Make a ground truth dict
ground_truth = {}

# Set all unique genres
all_genres = set(g for genres_list in df['genres'] for g in genres_list)

for genre in all_genres:
    # Filter all films with the current genre
    df_genre = df[df['genres'].apply(lambda x: genre in x)]
    
    # Sort by top 5 most popular
    df_genre_top5 = df_genre.sort_values(by='popularity', ascending=False).head(6)
    
    for idx, row in df_genre_top5.iterrows():
        film_id = row['id']
        relevant_films = df_genre_top5[df_genre_top5['id'] != film_id]['id'].tolist()
        ground_truth[film_id] = relevant_films
```

Karena ground truth masih menyimpan ID film, maka selanjutnya akan diubah menjadi judul film sesuai ID

```python
# Convert ground truth ID to movie title
ground_truth_titles = {}
for film_id, gt_ids in ground_truth.items():
  # Movie title
  main_title = df[df['id'] == int(film_id)]['original_title'].values[0]
    
  # Movie title on ground truth
  gt_titles = df[df['id'].isin(gt_ids)]['original_title'].tolist()
    
  ground_truth_titles[main_title] = gt_titles

print(ground_truth_titles)
```

Diperoleh hasil ground truth untuk film **Avatar**, yaitu:

- Pirates of the Caribbean: At World's End
- Batman v Superman: Dawn of Justice
- Pirates of the Caribbean: Dead Man's Chest
- Pirates of the Caribbean: The Curse of the Black Pearl
- Teenage Mutant Ninja Turtles

2. Evaluasi Menggunakan Metrik Ranking
Sistem rekomendasi dievaluasi menggunakan dua metrik ranking populer:

a. Mean Reciprocal Rank (MRR)
MRR menilai seberapa cepat sistem menemukan item yang relevan dalam daftar hasil.<br>
	$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Keterangan:

- $$|Q|$$: Jumlah query.
- $$\text{rank}_i$$: Posisi (peringkat) item relevan pertama dalam daftar hasil ke- $$i$$

b. Normalized Discounted Cumulative Gain (nDCG)<br>
Metrik ini mempertimbangkan posisi item relevan dalam urutan rekomendasi.<br>

- 	$$\text{DCG}_k = \sum_{i=1}{k} \frac{\text{rel}_i}{\log_2(i+1)}$$
  
- 	$$\text{nDCG}_k = \frac{\text{DCG}_k}{\text{IDCG}_k}$$

Keterangan:
- $$k$$: Jumlah item dalam daftar (top-5).
- $$\text{rel}_i$$: Skor relevansi item ke- $$i$$.
- $$\text{IDCG}_k$$: nilai DCG terbaik untuk $$k$$ item, digunakan untuk normalisasi.

Hasil Evaluasi Metrik (Top-5):

| Pendekatan | MRR | nDCG@5 |
| --- | --- | --- |
|TF-IDF - Overview | 0.00 |0.00 |
|TF-IDF - Genres |0.00 | 0.00 |
|BERT - Overview | 0.00 | 0.00 |
|BERT - Genres | 0.00 | 0.00 |

Nilai 0 menunjukkan bahwa tidak ada satu pun film dari ground truth yang muncul di daftar Top-5 hasil rekomendasi dari keempat pendekatan.

3. Evaluasi Manual
Meskipun nilai metrik MRR dan nDCG menunjukkan hasil 0, dilakukan evaluasi manual untuk melihat sejauh mana hasil rekomendasi mencerminkan kemiripan tematik dengan Avatar, menggunakan dua pendekatan:

a. Evaluasi terhadap Ground Truth
  - Tidak ada satu pun film dari ground truth yang muncul dalam daftar Top-5 rekomendasi dari keempat pendekatan.
  - Film seperti *Pirates of the Caribbean* dan *Teenage Mutant Ninja Turtles* sebenarnya memiliki sejumlah kemiripan dengan Avatar dalam hal:
    - Aksi dan petualangan berskala besar
    - Dunia fiksi yang kompleks dan penuh karakter ikonik
    - Unsur visual yang dominan (CGI-heavy)
  - Namun, sistem tidak berhasil menangkap asosiasi ini karena:
    - Perbedaan konteks naratif (*Avatar* lebih sci-fi, sedangkan POTC dan TMNT lebih fantasi atau superhero)
    - Keterbatasan representasi semantik dan kosakata dalam overview atau genre

b. Inspeksi Tematik (Visual Review)
[Hasil Rekomendasi Setiap Skema](img/Screenshot 2025-06-01 092424.png)
Beberapa film yang muncul di daftar rekomendasi memiliki kesamaan dengan Avatar dari sisi:

- Dunia asing dan visualisasi epik (misalnya: Jupiter Ascending)
- Peperangan antaraspek budaya atau spesies
- Karakter protagonis yang bertransformasi dalam dunia yang asing

Namun, sistem juga banyak merekomendasikan film dengan hubungan longgar atau bahkan tidak relevan secara tematik, terutama pada pendekatan TF-IDF + genres.

## Conclusion

### Kesimpulan Setiap Skema

Kesimpulan dari setiap skema sebagai berikut:

**1. TF-IDF Overview**

- Hasil rekomendasi:
	Apollo 18, Tears of the Sun, The American, Obitaemyy Ostrov, The Matrix
- Kelebihan:
	Cenderung menangkap kata-kata umum dari sinopsis
- Kekurangan:
	- Beberapa film seperti Semi-Pro atau The Adventures of Pluto Nash tidak relevan secara tematik/sci-fi dibanding Avatar. 
	- Hanya cocok secara keyword, bukan secara konteks cerita.
- Kesimpulan:
  Rekomendasi ini kurang masuk akal, dibuktikan dengan nilai cosine similarity yang jauh lebih rendah daripada skema lain.

**2. TF-IDF Genres**

- Hasil rekomendasi:
	Superman Returns, Man of Steel, X-Men: Days of Future Past, Jupiter Ascending, The Wolverine
- Kelebihan:
  - Relevansi genre cukup tinggi: film superhero, sci-fi, action
  - Banyak film dengan elemen futuristik atau pahlawan
- Kesimpulan:
	Rekomendasi ini lebih masuk akal dibanding overview-nya, dibuktikan dengan nilai cosine similarity yang banyak menyentuh angka 1 (sangat mirip).

**3. BERT Overview**

- Hasil rekomendasi:
  Alien: Resurrection, The Black Hole, Serenity, Aliens, Supernova
- Kelebihan:
  - Lebih semantik, karena BERT memahami konteks dan nuansa kalimat
  - Sebagian besar film bertema luar angkasa, sci-fi, alien
  - Hampir semua relevan dengan "Avatar" (futuristik, alien world, eksplorasi)
- Kesimpulan:
	Ini salah satu skema paling cocok secara tematik dan gaya cerita.

**4. BERT Genres**

- Hasil rekomendasi: 
	X-Men: Days of Future Past, Man of Steel, Superman, Beastmaster 2: Through the Portal of Time, Superman II
- Kelebihan:
  - Genre cocok, sci-fi/fantasy/action
  - Meskipun ada pengulangan franchise superhero yang bisa terlalu dominan
- Kesimpulan:
  Cukup baik, tapi mirip dengan TF-IDF genres

### Kesimpulan Keseluruhan

Berdasarkan hasil rekomendasi dari keempat pendekatan content-based filtering terhadap film *Avatar*, terlihat bahwa setiap skema menghasilkan pola rekomendasi yang berbeda. Pendekatan **TF-IDF** pada `overview` cenderung merekomendasikan film dengan kemiripan secara literal dalam teks sinopsis, namun tidak selalu relevan secara tematik — misalnya *Apollo 18* atau *The American*, yang memiliki kata kunci serupa namun latar cerita sangat berbeda. Sebaliknya, **TF-IDF** pada `genres` menunjukkan hasil yang sedikit lebih baik, dengan munculnya film-film superhero dan fiksi ilmiah seperti *Man of Steel*, *X-Men: Days of Future Past*, dan *Jupiter Ascending*, meskipun tetap belum menyentuh film dalam ground truth. Pendekatan **BERT** pada `overview` memberikan hasil yang lebih semantik dan relevan, seperti *Aliens*, *Serenity*, dan A*lien: Resurrection*, yang memiliki kemiripan konteks dunia luar angkasa, spesies asing, dan konflik manusia-alien—tema yang sangat dekat dengan *Avatar*. Terakhir, **BERT** pada `genres` juga merekomendasikan film superhero dan sci-fi klasik, seperti *Superman*, *X-Men*, dan *Superman II*, namun kembali terbatas oleh input genre yang sangat singkat.
Secara keseluruhan, pendekatan BERT dengan `overview` terlihat paling mendekati konteks naratif dan atmosfer Avatar, meskipun masih belum berhasil mencocokkan film yang ada dalam ground truth secara eksplisit. Meskipun nilai cosine similarity pada BERT dengan `overview` tidak sebagus TF-IDF dengan `genres` maupun BERT dengan `genres`, namun BERT dengan `overview` menunjukkan kemampuan untuk memahami konteks dan nuansa teks yang lebih dalam, sehingga memberikan rekomendasi yang lebih relevan.

## References

[1] Lops, P., De Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In Recommender Systems Handbook (pp. 73-105). Springer.
[2] Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. ACM Computing Surveys (CSUR), 52(1), 1–38. https://doi.org/10.1145/3285029

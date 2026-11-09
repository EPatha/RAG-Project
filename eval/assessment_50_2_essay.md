# Evaluasi Materi Generative AI dan RAG

## Pilihan/short answer (50 soal)

1. Apa yang dimaksud dengan Generative AI?
2. Sebutkan tiga modalitas yang dapat dihasilkan model generatif.
3. Apa perbedaan use-case generasi teks dan klasifikasi?
4. Mengapa privasi perlu dipertimbangkan sebelum memakai model hosted?
5. Apa fungsi tokenisasi pada LLM?
6. Apa peran attention pada Transformer?
7. Mengapa LLM dapat menghasilkan jawaban yang terdengar benar tetapi salah?
8. Apa pengaruh temperature terhadap variasi output?
9. Apa fungsi system prompt?
10. Mengapa konteks model memiliki batas token?
11. Apa fungsi API key pada OpenRouter?
12. Mengapa API key tidak boleh di-commit ke Git?
13. Apa arti endpoint OpenAI-compatible?
14. Sebutkan dua informasi yang perlu dicatat saat memilih model hosted.
15. Apa itu prompt engineering?
16. Sebutkan tiga komponen prompt yang dapat diuji.
17. Mengapa format output perlu ditentukan secara eksplisit?
18. Apa tujuan instruksi refusal pada prompt RAG?
19. Apa yang dimaksud prompt injection?
20. Mengapa output LLM tetap perlu divalidasi?
21. Jelaskan secara singkat proses forward diffusion.
22. Apa yang dipelajari model pada proses denoising?
23. Sebutkan dua parameter sampling diffusion.
24. Apa fungsi text encoder pada text-to-image?
25. Mengapa checkpoint diffusion perlu diperiksa lisensinya?
26. Apa yang dimaksud sistem multimodal?
27. Sebutkan dua risiko khusus pada input gambar.
28. Apa fungsi OCR dalam pipeline multimodal?
29. Apa itu Retrieval-Augmented Generation?
30. Sebutkan tahapan utama pipeline RAG project ini.
31. Mengapa dokumen harus dipecah menjadi chunk?
32. Apa fungsi metadata halaman pada hasil retrieval?
33. Apa karakteristik utama BM25?
34. Apa perbedaan dense retrieval dan lexical retrieval?
35. Apa tujuan hybrid retrieval?
36. Apa fungsi RRF dalam penggabungan hasil?
37. Apa fungsi reranker?
38. Mengapa RAG tidak otomatis menghilangkan semua halusinasi?
39. Apa yang dimaksud pertanyaan unanswerable?
40. Mengapa baseline extractive berguna?
41. Apa arti metrik Hit@k?
42. Apa arti MRR?
43. Apa tujuan nDCG?
44. Apa perbedaan recall dan precision pada retrieval?
45. Apa yang diukur lexical faithfulness?
46. Apa arti false-refusal rate?
47. Mengapa latency perlu diukur?
48. Mengapa evaluasi harus memisahkan answerable dan unanswerable?
49. Apa maksud iterasi satu variabel pada eksperimen RAG?
50. Sebutkan dua praktik untuk menjaga sistem RAG dapat diaudit.

## Kunci jawaban ringkas

1. Sistem yang menghasilkan konten baru dari pola data yang dipelajari.
2. Teks, gambar, audio, atau video.
3. Generasi membuat konten; klasifikasi memilih label.
4. Data dapat tersimpan atau diproses pihak ketiga.
5. Mengubah teks menjadi unit yang dapat diproses model.
6. Menimbang hubungan antar-token.
7. Model mengoptimalkan kelanjutan token, bukan kebenaran faktual.
8. Temperature tinggi memberi output lebih bervariasi.
9. Menetapkan perilaku dan batasan global model.
10. Memori komputasi dan biaya attention terbatas.
11. Mengautentikasi permintaan ke layanan.
12. Agar secret tidak tersebar.
13. Endpoint memakai pola request/response chat yang serupa.
14. Kualitas, biaya, latency, privasi, atau lisensi.
15. Rekayasa instruksi agar output konsisten dan dapat diuji.
16. Tujuan, konteks, format, batasan, atau contoh.
17. Agar output mudah divalidasi dan dipakai program.
18. Mencegah model mengarang saat bukti tidak tersedia.
19. Upaya memanipulasi instruksi model melalui input.
20. Model bisa salah, tidak lengkap, atau tidak mengikuti format.
21. Menambahkan noise bertahap pada data bersih.
22. Cara mengembalikan noise menjadi sampel bermakna.
23. Seed, steps, guidance scale, atau resolusi.
24. Mengubah teks menjadi representasi kondisi gambar.
25. Agar penggunaan model legal dan bertanggung jawab.
26. Sistem yang memproses lebih dari satu jenis data.
27. OCR salah, gambar buram, atau metadata sensitif.
28. Mengubah tulisan dalam gambar menjadi teks.
29. Generasi yang didahului pencarian sumber eksternal.
30. Ingest, chunk, retrieve, generate, dan tampilkan sumber.
31. Agar retrieval dapat memilih konteks yang relevan.
32. Menunjukkan asal bukti dan membantu audit.
33. Pencarian lexical berbasis frekuensi istilah dan panjang dokumen.
34. Dense memakai kemiripan vektor; lexical mencocokkan istilah.
35. Menggabungkan kekuatan pencocokan istilah dan makna.
36. Menggabungkan peringkat dari beberapa retriever.
37. Menyusun ulang kandidat berdasarkan relevansi yang lebih tajam.
38. Retrieval bisa salah dan model bisa salah menafsirkan konteks.
39. Pertanyaan yang buktinya tidak ada di corpus.
40. Menjadi smoke test tanpa ketergantungan LLM.
41. Apakah ada hasil relevan pada k posisi teratas.
42. Rata-rata reciprocal rank hasil relevan pertama.
43. Mengukur kualitas peringkat dengan diskonto posisi.
44. Recall mengukur temuan relevan; precision mengukur ketepatan hasil.
45. Seberapa banyak klaim jawaban didukung token konteks.
46. Proporsi pertanyaan yang sebenarnya dapat dijawab tetapi ditolak.
47. Untuk memastikan sistem memenuhi kebutuhan pengguna.
48. Agar kemampuan menjawab dan kemampuan menolak diukur terpisah.
49. Mengubah satu faktor agar penyebab perubahan dapat diidentifikasi.
50. Simpan sumber/sitasi dan catat konfigurasi serta hasil eksperimen.

## Essay (2 soal)

### Essay 1
Jelaskan rancangan sistem RAG untuk peraturan akademik dari ingestion sampai
evaluasi. Bandingkan BM25, dense retrieval, hybrid retrieval, dan reranking.
Sertakan strategi menangani pertanyaan yang tidak memiliki jawaban di corpus.

### Essay 2
Analisis perbedaan Ollama lokal dan OpenRouter hosted untuk aplikasi LLM.
Bahas privasi, biaya, latency, kualitas model, pengelolaan API key, dan
strategi evaluasi sebelum sistem digunakan oleh pengguna.

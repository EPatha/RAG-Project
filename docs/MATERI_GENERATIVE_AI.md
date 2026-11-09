# Materi Generative AI dan RAG

Dokumen ini melengkapi demo kode dengan ringkasan materi kuliah, peta use-case,
dan langkah praktik. Contoh peraturan di `data/sample/` adalah fiktif.

## 1. Lanskap Generative AI dan peta use-case

Generative AI menghasilkan teks, kode, gambar, audio, atau video berdasarkan
pola yang dipelajari dari data. Pemilihan use-case sebaiknya mempertimbangkan
nilai bisnis, risiko kesalahan, privasi, biaya, latency, dan kebutuhan audit.

| Kebutuhan | Teknologi yang sesuai | Contoh |
|---|---|---|
| Menjawab dokumen internal | RAG | Asisten peraturan akademik |
| Menulis atau meringkas | LLM | Ringkasan laporan |
| Membuat gambar | Diffusion model | Concept art |
| Memahami gambar + teks | Multimodal model | Tanya jawab dari foto |
| Otomasi terstruktur | LLM + tools | Ekstraksi JSON dan workflow |

## 2. Cara kerja LLM

LLM memecah input menjadi token, mengubahnya menjadi vektor, lalu memakai
arsitektur Transformer (attention) untuk memprediksi token berikutnya.
Output bersifat probabilistik, sehingga model dapat terdengar meyakinkan
meskipun salah. Temperature, konteks, system prompt, dan batas token
mempengaruhi perilaku. LLM tidak otomatis mengetahui data privat atau fakta
terbaru; sumber eksternal harus diberikan melalui RAG atau tools.

### Onboarding OpenRouter

OpenRouter menyediakan endpoint yang kompatibel dengan pola
`/chat/completions` dan meneruskan permintaan ke berbagai model. Jangan
menyimpan API key di Git.

```bash
cp .env.example .env
export OPENROUTER_API_KEY="isi_key_anda"
.venv/bin/python app.py --docs data/sample --retriever bm25 \
  --llm openrouter --openrouter-model openai/gpt-4o-mini
```

Integrasi berada di `rag/generators.py`. Kode akan gagal secara eksplisit jika
`OPENROUTER_API_KEY` belum tersedia.

## 3. Prompt engineering sebagai rekayasa

Prompt yang baik memiliki kontrak yang dapat diuji:

1. tetapkan peran dan tujuan;
2. berikan konteks yang relevan;
3. batasi sumber kebenaran;
4. tentukan format output;
5. nyatakan perilaku jika informasi tidak tersedia;
6. uji dengan contoh normal, ambigu, dan adversarial.

System prompt RAG pada `rag/generators.py` mewajibkan jawaban berdasarkan
konteks, sitasi `[nomor]`, Bahasa Indonesia, dan refusal saat konteks tidak
mendukung jawaban. Prompt bukan pengganti validasi: output tetap perlu
dievaluasi dengan metrik dan pemeriksaan manusia.

## 4. Model difusi

Model difusi membuat data dengan mempelajari proses denoising: data bersih
secara bertahap diberi noise saat training, kemudian model belajar membalik
proses tersebut saat sampling. Text-to-image biasanya memakai text encoder,
latent diffusion, U-Net/transformer denoiser, dan scheduler. Parameter penting
meliputi seed, steps, guidance scale, dan resolusi.

Contoh konseptual dan kode opsional tersedia di `examples/diffusion_demo.py`.
Dependensi diffusion sengaja tidak dipasang pada instalasi RAG dasar karena
ukurannya besar dan memerlukan model checkpoint terpisah.

## 5. Multimodal

Multimodal menggabungkan lebih dari satu modalitas, misalnya teks dan gambar.
Pipeline umum: ekstraksi/normalisasi input, encoder per modalitas, alignment
ke ruang representasi bersama, lalu generator multimodal. Risiko tambahan
meliputi OCR keliru, gambar buram, metadata sensitif, dan prompt injection
melalui teks di dalam gambar.

`examples/multimodal_rag.py` menunjukkan kontrak adapter sederhana untuk
mengubah halaman PDF menjadi image dan mengirimkannya ke model vision yang
kompatibel OpenRouter. Ini adalah contoh terpisah; pipeline teks utama tetap
deterministik dan tidak memerlukan model vision.

## 6. Keterbatasan model dan pengantar RAG

RAG mengurangi halusinasi dengan mengambil potongan dokumen yang relevan
sebelum generasi. Pipeline project ini adalah:

`dokumen -> parsing -> chunking -> BM25/dense -> hybrid/RRF -> generator -> sitasi`

RAG tetap memiliki keterbatasan: retrieval dapat melewatkan bukti, chunking
dapat memutus konteks, dokumen dapat kedaluwarsa, dan model dapat salah
menafsirkan konteks. Karena itu jawaban harus menampilkan sumber dan kasus
unanswerable harus diuji.

## 7. Praktik membangun RAG

1. definisikan corpus dan kebijakan sumber;
2. ingest PDF/TXT dan pertahankan metadata halaman;
3. bandingkan ukuran chunk dan overlap;
4. ukur BM25, dense, hybrid, dan reranker;
5. gunakan generator extractive untuk smoke test;
6. gunakan Ollama atau OpenRouter setelah retrieval tervalidasi;
7. tampilkan jawaban dan sumber di UI.

## 8. Evaluasi dan iterasi

`evaluate.py` menyediakan Hit@k, MRR, nDCG, Precision, token F1, recall,
ROUGE-L, lexical faithfulness, refusal accuracy, latency, dan optional
LLM-as-judge. Iterasi yang sehat mengubah satu variabel, menyimpan hasil,
membandingkan baseline, lalu memeriksa regresi pada pertanyaan yang gagal.

```bash
.venv/bin/python evaluate.py --docs data/sample \
  --qa eval/qa_sample.jsonl --generate --llm extractive --retrievers bm25
```

## 9. Etika dan keamanan

Anonimkan data sensitif, jangan commit secret, validasi output terstruktur,
catat sumber, dan sediakan human review untuk keputusan berdampak tinggi.
Gunakan hanya dokumen dan model yang lisensinya sesuai.

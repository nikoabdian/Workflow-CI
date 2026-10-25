# Workflow-CI

Kriteria 3 submission proyek akhir kelas **Membangun Sistem Machine Learning** — Dicoding Indonesia.

Repositori ini membungkus pelatihan model **Telco Customer Churn** sebagai
**MLflow Project**, lalu menjalankannya ulang secara otomatis lewat GitHub
Actions setiap kali trigger terpantik.

## Struktur repositori

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                     # workflow CI
├── MLProject/
│   ├── MLProject                      # definisi MLflow Project
│   ├── conda.yaml                     # environment (Python 3.12.7)
│   ├── modelling.py                   # skrip pelatihan
│   └── telco_preprocessing/           # dataset siap latih
│       ├── telco_clean.csv
│       ├── train.csv
│       ├── test.csv
│       └── scaler.joblib
├── DockerHub.txt                      # tautan Docker Hub
└── README.md
```

## MLflow Project

`MLProject` mendefinisikan tujuh parameter yang dapat diubah tanpa menyentuh kode:

| Parameter | Default | Keterangan |
|---|---|---|
| `data_path` | `telco_preprocessing/telco_clean.csv` | Lokasi dataset |
| `n_estimators` | `200` | Jumlah pohon Random Forest |
| `max_depth` | `8` | Kedalaman maksimum (`0` = tanpa batas) |
| `min_samples_leaf` | `1` | Minimum sampel per daun |
| `class_weight` | `balanced` | Penyeimbang kelas |
| `test_size` | `0.2` | Proporsi data uji |
| `cv_folds` | `5` | Jumlah lipatan cross-validation |

Nilai default diambil dari hasil `GridSearchCV` pada Kriteria 2.

Menjalankan secara lokal:

```bash
mlflow run MLProject --env-manager=local
```

Dengan parameter khusus:

```bash
mlflow run MLProject --env-manager=local -P n_estimators=300 -P max_depth=12
```

## Workflow CI

`ci.yml` berjalan ketika:

- ada `push` ke `main` yang menyentuh `MLProject/`;
- ada `pull_request` ke `main`;
- setiap Senin pukul 02:00 UTC;
- dijalankan manual lewat **Actions → Run workflow** (dapat mengatur
  `n_estimators` dan `max_depth`).

### Tahapan

| # | Tahap | Level |
|---|---|---|
| 1 | Checkout dan pasang Python 3.12.7 + MLflow 2.19.0 | Basic |
| 2 | Periksa dataset hasil preprocessing | Basic |
| 3 | Jalankan `mlflow run MLProject` | Basic |
| 4 | Ambil Run ID dan tulis ringkasan metrik ke Job Summary | Basic |
| 5 | Unggah `mlruns/` dan artefak sebagai GitHub Actions artifact | Skilled |
| 6 | Commit artefak model ke folder `artefak_terbaru/` pada repo ini | Skilled |
| 7 | Bangun Docker image dengan `mlflow models build-docker` | Advanced |
| 8 | Dorong image ke Docker Hub dengan tiga tag | Advanced |

### Secrets yang diperlukan

Settings → Secrets and variables → Actions:

| Secret | Isi |
|---|---|
| `DOCKERHUB_USERNAME` | `abdianniko` |
| `DOCKERHUB_TOKEN` | Access token dari https://hub.docker.com/settings/security |

## Model yang dihasilkan

Random Forest dengan parameter hasil tuning Kriteria 2
(`class_weight='balanced'`, `max_depth=8`, `min_samples_leaf=1`, `n_estimators=200`):

| Metrik | Data uji |
|---|---|
| accuracy | 0,7673 |
| precision | 0,5439 |
| recall | 0,7500 |
| f1_score | 0,6305 |
| roc_auc | 0,8413 |
| cv_f1_mean | 0,6392 ± 0,0154 |

## Menjalankan model sebagai layanan

Lewat Docker Hub:

```bash
docker pull abdianniko/telco-churn-model:latest
docker run -p 5005:8080 abdianniko/telco-churn-model:latest
```

Lewat MLflow secara langsung:

```bash
mlflow models serve -m "runs:/<run_id>/model" -p 5005 --no-conda
```

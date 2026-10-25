"""Kriteria 3 — skrip pelatihan yang dijalankan sebagai MLflow Project.

Skrip ini adalah penyesuaian dari ``modelling.py`` dan ``modelling_tuning.py``
pada Kriteria 2 agar dapat dipanggil lewat ``mlflow run`` dan dijalankan ulang
secara otomatis oleh GitHub Actions. Seluruh parameter dibuat dapat diatur dari
berkas ``MLProject`` supaya re-training dapat dikonfigurasi tanpa mengubah kode.

Menjalankan lewat MLflow Project::

    mlflow run MLProject --env-manager=local

Menjalankan langsung::

    python modelling.py --data_path telco_preprocessing/telco_clean.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_COL = "Churn"
EXPERIMENT_NAME = "Telco Churn - CI Retraining"
ARTIFACT_DIR = "artefak_ci"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pelatihan model untuk MLflow Project.")
    parser.add_argument("--data_path", type=str, default="telco_preprocessing/telco_clean.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--class_weight", type=str, default="balanced")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv_folds", type=int, default=5)
    return parser.parse_args()


def muat_data(data_path: str, test_size: float):
    """Memuat data bersih dan membaginya dengan stratifikasi."""
    folder = os.path.dirname(data_path) or "."
    train_path = os.path.join(folder, "train.csv")
    test_path = os.path.join(folder, "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"Memuat pembagian data dari '{folder}'")
    else:
        df = pd.read_csv(data_path)
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
        )
        print(f"Memuat dan membagi ulang '{data_path}'")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    print(f"  Data latih : {X_train.shape}")
    print(f"  Data uji   : {X_test.shape}")
    return X_train, X_test, y_train, y_test


def simpan_artefak(model, X_train, y_test, y_pred, y_proba) -> None:
    """Membuat artefak pendukung: confusion matrix, ROC, feature importance, laporan."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    p = lambda nama: os.path.join(ARTIFACT_DIR, nama)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), nilai in np.ndenumerate(cm):
        ax.text(j, i, str(nilai), ha="center", va="center",
                color="white" if nilai > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["Tidak churn", "Churn"])
    ax.set_yticks([0, 1], ["Tidak churn", "Churn"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion matrix — data uji")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(p("confusion_matrix.png"), dpi=120)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, lw=2, color="#c44e52", label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Kurva ROC — data uji")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(p("roc_curve.png"), dpi=120)
    plt.close(fig)

    kepentingan = (
        pd.DataFrame({"fitur": X_train.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    kepentingan.to_csv(p("feature_importance.csv"), index=False)

    atas = kepentingan.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(atas["fitur"], atas["importance"], color="#55a868")
    ax.set_xlabel("Importance")
    ax.set_title("15 fitur paling berpengaruh")
    fig.tight_layout()
    fig.savefig(p("feature_importance.png"), dpi=120)
    plt.close(fig)

    with open(p("classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred,
                                      target_names=["Tidak churn", "Churn"], digits=4))


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("RE-TRAINING OTOMATIS - TELCO CUSTOMER CHURN")
    print("=" * 70)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    mlflow.sklearn.autolog(disable=True)

    # Saat dipanggil lewat `mlflow run`, MLflow Project sudah membuat run sendiri
    # dan menaruh ID-nya pada MLFLOW_RUN_ID. Run itu dipakai ulang agar tidak
    # bentrok. Saat skrip dipanggil langsung, run baru dibuat di sini.
    run_id_project = os.environ.get("MLFLOW_RUN_ID")
    if run_id_project:
        print(f"Memakai run yang dibuat MLflow Project: {run_id_project}")
        ctx = mlflow.start_run(run_id=run_id_project)
    else:
        mlflow.set_experiment(EXPERIMENT_NAME)
        ctx = mlflow.start_run(run_name="ci_retraining")

    X_train, X_test, y_train, y_test = muat_data(args.data_path, args.test_size)

    class_weight = None if args.class_weight.lower() in ("none", "null", "") else args.class_weight
    max_depth = None if args.max_depth <= 0 else args.max_depth

    with ctx as run:
        print(f"\nRun ID: {run.info.run_id}")

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        mulai = time.time()
        model.fit(X_train, y_train)
        durasi = time.time() - mulai

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        skor_cv = cross_val_score(
            model, X_train, y_train, scoring="f1",
            cv=StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1,
        )

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "max_depth": str(max_depth),
                "min_samples_leaf": args.min_samples_leaf,
                "class_weight": str(class_weight),
                "test_size": args.test_size,
                "cv_folds": args.cv_folds,
                "random_state": RANDOM_STATE,
                "model_type": "RandomForestClassifier",
                "jumlah_fitur": X_train.shape[1],
                "jumlah_data_latih": len(X_train),
                "jumlah_data_uji": len(X_test),
            }
        )

        metrik = {
            "training_accuracy_score": accuracy_score(y_train, model.predict(X_train)),
            "training_f1_score": f1_score(y_train, model.predict(X_train), zero_division=0),
            "test_accuracy_score": accuracy_score(y_test, y_pred),
            "test_precision_score": precision_score(y_test, y_pred, zero_division=0),
            "test_recall_score": recall_score(y_test, y_pred, zero_division=0),
            "test_f1_score": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "test_log_loss": log_loss(y_test, y_proba),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "test_average_precision": average_precision_score(y_test, y_proba),
            "test_matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
            "test_specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "test_true_negative": float(tn),
            "test_false_positive": float(fp),
            "test_false_negative": float(fn),
            "test_true_positive": float(tp),
            "cv_f1_mean": float(skor_cv.mean()),
            "cv_f1_std": float(skor_cv.std()),
            "training_duration_seconds": durasi,
        }
        mlflow.log_metrics(metrik)

        print("\nPerforma pada data uji:")
        for nama in ("test_accuracy_score", "test_precision_score", "test_recall_score",
                     "test_f1_score", "test_roc_auc"):
            print(f"  {nama:<24}: {metrik[nama]:.4f}")
        print(f"  cv_f1_mean              : {metrik['cv_f1_mean']:.4f} "
              f"(+/- {metrik['cv_f1_std']:.4f})")

        simpan_artefak(model, X_train, y_test, y_pred, y_proba)
        mlflow.log_artifacts(ARTIFACT_DIR, artifact_path="artefak_ci")

        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )

        mlflow.set_tags(
            {
                "dataset": "Telco Customer Churn",
                "tahap": "CI retraining",
                "kriteria": "3 - Workflow CI",
                "pemicu": os.environ.get("GITHUB_EVENT_NAME", "manual"),
                "commit": os.environ.get("GITHUB_SHA", "lokal")[:8],
            }
        )

        # Run ID dibutuhkan langkah berikutnya pada GitHub Actions untuk
        # membangun Docker image dari model yang baru dilatih.
        run_id = run.info.run_id
        with open("run_id.txt", "w", encoding="utf-8") as f:
            f.write(run_id)

        ringkasan = {
            "run_id": run_id,
            "experiment": EXPERIMENT_NAME,
            "params": {
                "n_estimators": args.n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "class_weight": class_weight,
            },
            "metrics": {k: float(v) for k, v in metrik.items()},
        }
        with open("metric_info.json", "w", encoding="utf-8") as f:
            json.dump(ringkasan, f, indent=2)

        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as f:
                f.write(f"run_id={run_id}\n")

        print(f"\nArtefak tersimpan di: {mlflow.get_artifact_uri()}")

    print("=" * 70)
    print("SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()

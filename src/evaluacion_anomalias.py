"""Métricas, visualizaciones y exportación de la comparativa de validación."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def evaluar_detector(resultado, y_valid, k=100):
    """AP es la métrica principal; top-k ilustra un presupuesto de revisión fijo."""
    y = np.asarray(y_valid)
    scores = np.asarray(resultado["scores"])
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("Los scores deben ser finitos y corresponder a las etiquetas.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("La evaluación requiere etiquetas 0 y 1.")
    k = min(k, len(y))
    # En empates se conserva el orden de validación, sin consultar las etiquetas.
    posiciones = np.argsort(-scores, kind="stable")[:k]
    encontrados = int(y[posiciones].sum())
    return {"Modelo": resultado["nombre"],
            "AP validación": average_precision_score(y, scores),
            "ROC AUC": roc_auc_score(y, scores),
            "Revisadas": k, "Fraudes top-k": encontrados,
            "Precisión top-k": encontrados / k,
            "Recall top-k": encontrados / y.sum(),
            "Entrenamiento (s)": resultado["segundos_entrenamiento"],
            "Scores (s)": resultado["segundos_score"]}


def graficar_detector(resultado, y_valid):
    """Muestra la curva PR y los scores por clase en validación."""
    y = np.asarray(y_valid)
    scores = resultado["scores"]
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].step(recall, precision, where="post", label=f"AP = {ap:.4f}")
    axes[0].axhline(y.mean(), linestyle="--", color="gray", label="Prevalencia")
    axes[0].set(xlabel="Recall", ylabel="Precisión", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend()
    axes[1].boxplot([scores[y == 0], scores[y == 1]], showfliers=False)
    axes[1].set_xticks([1, 2], ["Normal", "Fraude"])
    axes[1].set(ylabel="Score de anomalía", title="Mediana y dispersión (sin extremos)")
    fig.suptitle(resultado["nombre"] + " — validación")
    fig.tight_layout()
    return fig


def comparar_detectores(resultados, y_valid):
    tabla = pd.DataFrame([evaluar_detector(r, y_valid) for r in resultados.values()])
    return tabla.sort_values("AP validación", ascending=False).reset_index(drop=True)


def graficar_comparacion(resultados, y_valid):
    fig, ax = plt.subplots(figsize=(9, 6))
    for nombre, resultado in resultados.items():
        precision, recall, _ = precision_recall_curve(y_valid, resultado["scores"])
        ap = average_precision_score(y_valid, resultado["scores"])
        ax.step(recall, precision, where="post", label=f"{nombre}: AP {ap:.4f}")
    ax.axhline(np.mean(y_valid), color="gray", linestyle="--", label="Prevalencia")
    ax.set(xlabel="Recall", ylabel="Precisión", title="Comparativa sobre la misma validación",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    return fig


def guardar_comparacion(resultados, y_valid, indices_entrenamiento, semilla, carpeta):
    """Guarda métricas, scores identificados por fila y configuración reproducible."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    comparar_detectores(resultados, y_valid).to_csv(carpeta / "comparacion_anomalias.csv", index=False)
    scores = pd.DataFrame({"fila_original": y_valid.index, "Class": y_valid.to_numpy()})
    for nombre, resultado in resultados.items():
        scores[nombre] = resultado["scores"]
    scores.to_csv(carpeta / "scores_validacion.csv", index=False)
    pd.DataFrame({"fila_original": indices_entrenamiento}).to_csv(
        carpeta / "muestra_entrenamiento_anomalias.csv", index=False)
    configuracion = {"semilla": semilla, "filas_entrenamiento": len(indices_entrenamiento),
                     "entrenamiento": "Solo normales, misma muestra para todos",
                     "preprocesamiento": "StandardScaler ajustado solo a la muestra de entrenamiento",
                     "modelos": {n: r["modelo"].get_params() for n, r in resultados.items()}}
    (carpeta / "configuracion_anomalias.json").write_text(
        json.dumps(configuracion, indent=2, ensure_ascii=False), encoding="utf-8")

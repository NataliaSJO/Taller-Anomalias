"""Detectores del material de clase adaptados a transacciones tabulares."""

from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def preparar_anomalias(X_train, y_train, X_valid, max_normales=20000, semilla=42):
    """Selecciona normales solo de entrenamiento y ajusta allí el escalador."""
    if not X_train.index.equals(y_train.index):
        raise ValueError("Las entradas y etiquetas de entrenamiento deben estar alineadas.")
    normales = X_train.loc[y_train.eq(0)]
    normales = normales.sample(n=min(max_normales, len(normales)), random_state=semilla)
    escalador = StandardScaler()
    Z_train = escalador.fit_transform(normales)
    Z_valid = escalador.transform(X_valid)
    return Z_train, Z_valid, escalador, normales.index.copy()


def crear_modelo(nombre, semilla=42):
    """Configuraciones iniciales fijas; no se han optimizado sobre validación."""
    modelos = {
        "Mahalanobis": LedoitWolf(),
        "PCA": PCA(n_components=10, svd_solver="full"),
        "Isolation Forest": IsolationForest(n_estimators=200, max_samples=256,
                                            random_state=semilla, n_jobs=-1),
        "GMM": GaussianMixture(n_components=4, covariance_type="full",
                               reg_covar=1e-5, max_iter=200, n_init=1,
                               random_state=semilla),
        "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1),
        "K-means": KMeans(n_clusters=8, n_init=10, random_state=semilla),
    }
    if nombre not in modelos:
        raise ValueError(f"Modelo desconocido: {nombre}")
    return modelos[nombre]


def calcular_scores(nombre, modelo, X):
    """Todos los scores crecen con la anomalía; no son probabilidades de fraude."""
    if nombre == "Mahalanobis":
        return modelo.mahalanobis(X)  # Distancia de Mahalanobis al cuadrado.
    if nombre == "PCA":
        reconstruccion = modelo.inverse_transform(modelo.transform(X))
        return np.mean((X - reconstruccion) ** 2, axis=1)
    if nombre == "K-means":
        return modelo.transform(X).min(axis=1)
    if nombre in ["Isolation Forest", "GMM", "LOF"]:
        return -modelo.score_samples(X)
    raise ValueError(f"Modelo desconocido: {nombre}")


def entrenar_detector(nombre, X_train, X_valid, semilla=42):
    """Entrena y puntúa datos nuevos, sin recibir etiquetas de validación."""
    modelo = crear_modelo(nombre, semilla)
    inicio = perf_counter()
    modelo.fit(X_train)
    tiempo_entrenamiento = perf_counter() - inicio
    if nombre == "GMM" and not modelo.converged_:
        raise RuntimeError("GMM no ha convergido; revisar el ajuste antes de comparar.")
    inicio = perf_counter()
    scores = calcular_scores(nombre, modelo, X_valid)
    tiempo_score = perf_counter() - inicio
    if scores.shape != (len(X_valid),) or not np.isfinite(scores).all():
        raise ValueError("El detector debe producir un score finito por fila.")
    return {"nombre": nombre, "modelo": modelo, "scores": scores,
            "segundos_entrenamiento": tiempo_entrenamiento,
            "segundos_score": tiempo_score}

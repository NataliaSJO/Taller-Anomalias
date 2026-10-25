"""Tablas y gráficos descriptivos para entender los datos."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resumen_datasets(construccion, produccion):
    """Cuenta duplicados posteriores a la primera aparición, como duplicated()."""
    filas = []
    for nombre, datos in [("Construcción", construccion), ("Producción", produccion)]:
        variables = datos.drop(columns="Class")
        filas.append({
            "Dataset": nombre,
            "Filas": len(datos),
            "Variables de entrada": variables.shape[1],
            "Ausentes en variables": int(variables.isna().sum().sum()),
            "Etiquetas ausentes": int(datos["Class"].isna().sum()),
            "Filas repetidas completas": int(datos.duplicated().sum()),
            "Filas repetidas por variables": int(variables.duplicated().sum()),
        })
    return pd.DataFrame(filas).set_index("Dataset")


def tabla_clases(datos):
    conteo = datos["Class"].value_counts().reindex([0, 1], fill_value=0)
    tabla = pd.DataFrame({"Transacciones": conteo, "Porcentaje": 100 * conteo / len(datos)})
    tabla.index = ["Normal (0)", "Fraude (1)"]
    return tabla


def grafico_clases(datos):
    tabla = tabla_clases(datos)
    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.bar(tabla.index, tabla["Transacciones"], color=["#377eb8", "#e6550d"])
    ax.set_yscale("log")
    ax.set_ylabel("Número de transacciones (escala logarítmica)")
    ax.set_title("Clases en construcción: muy pocos fraudes")
    ax.bar_label(barras, labels=[f"{n:,}" for n in tabla["Transacciones"]], padding=4)
    ax.set_ylim(1, tabla["Transacciones"].max() * 3)
    fig.tight_layout()
    return fig


def resumen_importes(datos):
    tabla = datos.groupby("Class")["Amount"].describe(percentiles=[0.5, 0.9, 0.99])
    return tabla.rename(index={0: "Normal", 1: "Fraude"})


def graficos_variables(datos, columnas=("Amount", "X1", "X2", "X3")):
    """Compara clases en entrenamiento con bins comunes y densidades normalizadas."""
    fig, ejes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, columna in zip(ejes.flat, columnas):
        valores = datos[columna]
        if columna == "Amount":
            valores = np.log1p(valores)
        bins = np.histogram_bin_edges(valores, bins=45)
        for clase, nombre, color in [(0, "Normal", "#377eb8"), (1, "Fraude", "#e6550d")]:
            ax.hist(valores[datos["Class"] == clase], bins=bins, density=True,
                    histtype="step", linewidth=1.8, label=nombre, color=color)
        ax.set_xlabel("log(1 + Amount)" if columna == "Amount" else columna)
        ax.set_ylabel("Densidad dentro de cada clase")
        ax.legend()
    fig.suptitle("Distribuciones de entrenamiento: normal frente a fraude")
    fig.tight_layout()
    return fig

"""Carga y comprobaciones básicas, sin modificar las transacciones."""

from pathlib import Path

import numpy as np
import pandas as pd


def cargar_datos(carpeta="datos"):
    """Devuelve construcción y producción conservando su orden original."""
    carpeta = Path(carpeta)
    construccion = pd.read_csv(carpeta / "credit_card_construction.csv")
    produccion = pd.read_csv(carpeta / "credit_card_production.csv")
    columnas = [f"X{i}" for i in range(1, 21)] + ["Amount", "Class"]
    for nombre, datos in [("construcción", construccion), ("producción", produccion)]:
        if list(datos.columns) != columnas:
            raise ValueError(f"Columnas inesperadas en {nombre}.")
        if not np.isfinite(datos.drop(columns="Class").to_numpy()).all():
            raise ValueError(f"Hay variables ausentes o infinitas en {nombre}.")
    if not construccion["Class"].isin([0, 1]).all():
        raise ValueError("Construcción debe tener etiquetas 0 o 1 en todas las filas.")
    if not produccion["Class"].isna().all():
        raise ValueError("Se esperaba que las etiquetas de producción estuvieran vacías.")
    return construccion, produccion

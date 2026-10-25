"""Separación reproducible, aproximadamente estratificada y sin duplicados cruzados."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def separar_datos(datos, semilla=42):
    """Agrupa variables idénticas; reparte cinco folds en 3/1/1 (60/20/20 aprox.).

    No elimina filas. La etiqueta se usa para estratificar, nunca para definir
    si dos transacciones tienen las mismas variables. Los índices son posiciones
    de fila, empezando en cero, en el CSV leído con pandas.
    """
    X = datos.drop(columns="Class")
    y = datos["Class"].astype(int)
    grupos, _ = pd.factorize(pd.MultiIndex.from_frame(X), sort=False)
    etiquetas = pd.DataFrame({"grupo": grupos, "clase": y.to_numpy()})
    if etiquetas.groupby("grupo")["clase"].nunique().gt(1).any():
        raise ValueError("Existen variables idénticas con etiquetas contradictorias; revisarlas primero.")

    separador = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
    asignacion = pd.DataFrame({"fila_original": range(len(datos)), "grupo": grupos})
    asignacion["particion"] = "entrenamiento"
    nombres = ["validacion", "prueba", "entrenamiento", "entrenamiento", "entrenamiento"]
    for nombre, (_, posiciones) in zip(nombres, separador.split(X, y, grupos)):
        asignacion.loc[posiciones, "particion"] = nombre

    # Comprobaciones ejecutables: cada fila y cada grupo pertenecen a una sola parte.
    assert asignacion["fila_original"].is_unique
    assert len(asignacion) == len(datos)
    assert asignacion.groupby("grupo")["particion"].nunique().eq(1).all()
    particiones = {}
    for nombre in ["entrenamiento", "validacion", "prueba"]:
        posiciones = asignacion.loc[asignacion["particion"] == nombre, "fila_original"]
        particiones[nombre] = datos.iloc[posiciones].copy()
        if particiones[nombre]["Class"].nunique() != 2:
            raise ValueError(f"La partición {nombre} no contiene las dos clases.")
    return particiones, asignacion


def resumen_particiones(particiones):
    total = sum(len(datos) for datos in particiones.values())
    filas = []
    for nombre, datos in particiones.items():
        fraudes = int(datos["Class"].sum())
        filas.append({"Partición": nombre, "Filas": len(datos),
                      "% del total": 100 * len(datos) / total,
                      "Normales": len(datos) - fraudes, "Fraudes": fraudes,
                      "% fraude": 100 * fraudes / len(datos)})
    return pd.DataFrame(filas).set_index("Partición")


def comprobar_separacion(asignacion):
    """Tabla de intersecciones; todas deben ser cero."""
    nombres = ["entrenamiento", "validacion", "prueba"]
    filas = []
    for i, izquierda in enumerate(nombres):
        for derecha in nombres[i + 1:]:
            a = asignacion[asignacion["particion"] == izquierda]
            b = asignacion[asignacion["particion"] == derecha]
            filas.append({"Comparación": f"{izquierda} / {derecha}",
                          "Filas compartidas": len(set(a.fila_original) & set(b.fila_original)),
                          "Grupos compartidos": len(set(a.grupo) & set(b.grupo))})
    return pd.DataFrame(filas).set_index("Comparación")


def guardar_asignacion(asignacion, ruta="resultados/particiones.csv"):
    """Guarda posiciones originales para reutilizar exactamente la separación."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    asignacion.to_csv(ruta, index=False)
    return ruta

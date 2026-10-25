"""Comprobaciones del contrato de separación, con datos sintéticos pequeños."""

import unittest

import pandas as pd

from src.separacion import comprobar_separacion, separar_datos


class SeparacionTest(unittest.TestCase):
    def setUp(self):
        # Cada transacción aparece dos veces y tiene un índice no consecutivo.
        originales = pd.DataFrame({"X1": range(100), "Amount": range(100),
                                   "Class": [0] * 75 + [1] * 25})
        self.datos = pd.concat([originales, originales], ignore_index=True)
        self.datos.index = self.datos.index * 3

    def test_conserva_filas_y_no_comparte_grupos(self):
        partes, asignacion = separar_datos(self.datos)
        self.assertEqual(sorted(asignacion.fila_original), list(range(len(self.datos))))
        self.assertEqual(sum(map(len, partes.values())), len(self.datos))
        self.assertTrue(comprobar_separacion(asignacion).eq(0).all().all())
        for nombre, parte in partes.items():
            posiciones = asignacion.loc[asignacion.particion == nombre, "fila_original"]
            pd.testing.assert_frame_equal(parte, self.datos.iloc[posiciones])

    def test_reproducible_y_no_modifica_entrada(self):
        original = self.datos.copy(deep=True)
        _, primera = separar_datos(self.datos, semilla=42)
        _, segunda = separar_datos(self.datos, semilla=42)
        pd.testing.assert_frame_equal(primera, segunda)
        pd.testing.assert_frame_equal(self.datos, original)

    def test_detecta_etiquetas_contradictorias(self):
        self.datos.iloc[100, self.datos.columns.get_loc("Class")] = 1
        with self.assertRaisesRegex(ValueError, "contradictorias"):
            separar_datos(self.datos)


if __name__ == "__main__":
    unittest.main()

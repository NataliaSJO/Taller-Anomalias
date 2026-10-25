"""Verifica orientación de scores y ausencia de ajuste con validación."""

import unittest

import numpy as np
import pandas as pd

from src.evaluacion_anomalias import evaluar_detector
from src.modelos_anomalias import preparar_anomalias, entrenar_detector


class AnomaliasTest(unittest.TestCase):
    def test_escalador_solo_aprende_normales_de_entrenamiento(self):
        X = pd.DataFrame({"a": [0., 2., 1000.], "b": [2., 4., 2000.]})
        y = pd.Series([0, 0, 1])
        valid = pd.DataFrame({"a": [9999.], "b": [9999.]})
        _, _, escalador, indices = preparar_anomalias(X, y, valid)
        np.testing.assert_allclose(escalador.mean_, [1, 3])
        self.assertEqual(set(indices), {0, 1})

    def test_detectores_asignan_mayor_score_a_extremos_nuevos(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(300, 21))
        valid = np.vstack([rng.normal(size=(20, 21)), rng.normal(30, 1, size=(20, 21))])
        for nombre in ["Mahalanobis", "PCA", "Isolation Forest", "GMM", "LOF", "K-means"]:
            with self.subTest(modelo=nombre):
                resultado = entrenar_detector(nombre, train, valid)
                scores = resultado["scores"]
                self.assertEqual(scores.shape, (40,))
                self.assertTrue(np.isfinite(scores).all())
                self.assertGreater(np.median(scores[20:]), np.median(scores[:20]))

    def test_metricas_de_ranking_con_resultado_conocido(self):
        resultado = {"nombre": "Ejemplo", "scores": np.array([0.1, 0.9, 0.8, 0.2]),
                     "segundos_entrenamiento": 0, "segundos_score": 0}
        fila = evaluar_detector(resultado, pd.Series([0, 1, 1, 0]), k=2)
        self.assertEqual(fila["AP validación"], 1)
        self.assertEqual(fila["Fraudes top-k"], 2)
        self.assertEqual(fila["Recall top-k"], 1)


if __name__ == "__main__":
    unittest.main()

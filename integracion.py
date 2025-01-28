import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class DataExtractor:
    """Clase para extraer estadísticas de jugadores desde SofaScore."""

    def __init__(self, scraper):
        self.scraper = scraper

    def scrape_player_match_stats_parallel(self, matches, jugador):
        """
        Obtiene estadísticas del jugador en paralelo para una lista de partidos.
        """
        def fetch_stats(match):
            try:
                match_id = match['id']
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                partido = f"{home_team} vs {away_team}"
                fecha_partido = datetime.fromtimestamp(match['startTimestamp']).strftime('%d/%m/%Y')
                jornada_id = match.get('roundInfo', {}).get('round', 0)

                # Obtener estadísticas del partido
                player_stats = self.scraper.scrape_player_match_stats(match_id)

                if 'name' in player_stats.columns:
                    jugador_stats = player_stats[player_stats['name'] == jugador]
                    if not jugador_stats.empty:
                        jugador_stats_filtered = {
                            'match_id': match_id,
                            'partido': partido,
                            'fecha': fecha_partido,
                            'jornada_id': jornada_id,
                            'player_id': jugador_stats['id'].values[0] if 'id' in jugador_stats.columns else None,
                            'player_name': jugador_stats['name'].values[0],
                            'goles': jugador_stats['goals'].values[0] if 'goals' in jugador_stats.columns else 0,
                            'asistencias': jugador_stats['goalAssist'].values[0] if 'goalAssist' in jugador_stats.columns else 0,
                            'goles_esperados': jugador_stats['expectedGoals'].values[0] if 'expectedGoals' in jugador_stats.columns else 0,
                            'oportunidades_claras_falladas': jugador_stats['bigChanceMissed'].values[0] if 'bigChanceMissed' in jugador_stats.columns else 0,
                            'tiros_a_puerta': jugador_stats['onTargetScoringAttempt'].values[0] if 'onTargetScoringAttempt' in jugador_stats.columns else 0,
                            'balones_recuperados': jugador_stats['totalClearance'].values[0] if 'totalClearance' in jugador_stats.columns else 0,
                            'minutos_jugados': jugador_stats['minutesPlayed'].values[0] if 'minutesPlayed' in jugador_stats.columns else 0,
                            'suplente': jugador_stats['substitute'].values[0] if 'substitute' in jugador_stats.columns else False,
                            'total_pases': jugador_stats['totalPass'].values[0] if 'totalPass' in jugador_stats.columns else 0,
                            'pases_exitosos': jugador_stats['accuratePass'].values[0] if 'accuratePass' in jugador_stats.columns else 0,
                            'total_pases_largos': jugador_stats['totalLongBalls'].values[0] if 'totalLongBalls' in jugador_stats.columns else 0,
                            'pases_largos_exitosos': jugador_stats['accurateLongBalls'].values[0] if 'accurateLongBalls' in jugador_stats.columns else 0,
                            'toques': jugador_stats['touches'].values[0] if 'touches' in jugador_stats.columns else 0,
                            'aereos_ganados': jugador_stats['aerialWon'].values[0] if 'aerialWon' in jugador_stats.columns else 0,
                            'duelos_perdidos': jugador_stats['duelLost'].values[0] if 'duelLost' in jugador_stats.columns else 0,
                            'duelos_ganados': jugador_stats['duelWon'].values[0] if 'duelWon' in jugador_stats.columns else 0,
                            'asistencias_esperadas': jugador_stats['expectedAssists'].values[0] if 'expectedAssists' in jugador_stats.columns else 0,
                            'recibio_faul': jugador_stats['wasFouled'].values[0] if 'wasFouled' in jugador_stats.columns else 0,
                            'aereos_perdidos': jugador_stats['aerialLost'].values[0] if 'aerialLost' in jugador_stats.columns else 0,
                            'pases_clave': jugador_stats['keyPass'].values[0] if 'keyPass' in jugador_stats.columns else 0,
                            'oportunidades_claras_creadas': jugador_stats['bigChanceCreated'].values[0] if 'bigChanceCreated' in jugador_stats.columns else 0,
                            'capitan': jugador_stats['captain'].values[0] if 'captain' in jugador_stats.columns else False,
                            'FuerasDeLugar': jugador_stats['totalOffside'].values[0] if 'totalOffside' in jugador_stats.columns else 0,
                            'marketValueCurrency': jugador_stats['marketValueCurrency'].values[0] if 'marketValueCurrency' in jugador_stats.columns else 0,
                            'rating_partido': jugador_stats['rating'].values[0] if 'rating' in jugador_stats.columns else None
                        }
                        return jugador_stats_filtered
            except Exception as e:
                print(f"Error obteniendo estadísticas para el partido {match_id}: {e}")
            return None

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_stats, match): match for match in matches}
            results = [future.result() for future in as_completed(futures)]

        return [res for res in results if res]

    def obtener_estadisticas_jugador_optimizado(self, jugador, league, season):
        """Obtiene estadísticas del jugador y devuelve un DataFrame."""
        matches = self.scraper.get_match_dicts(year=season, league=league)
        jugador_data = self.scrape_player_match_stats_parallel(matches, jugador)

        if not jugador_data:
            print(f"No se encontraron estadísticas para {jugador}.")
            return pd.DataFrame()

        jugador_df = pd.DataFrame(jugador_data)
        jugador_df.fillna(value=0, inplace=True)
        return jugador_df


class DataSaver:
    """Clase para guardar datos en un archivo CSV."""

    @staticmethod
    def guardar_estadisticas_csv(estadisticas, archivo_salida):
        """Guarda las estadísticas en un archivo CSV."""
        if estadisticas:
            df_combinado = pd.concat(estadisticas, ignore_index=True)
            df_combinado.to_csv(archivo_salida, index=False, encoding='utf-8')
            print(f"Estadísticas guardadas en {archivo_salida}")
        else:
            print("No hay estadísticas para guardar.")


class RatingPredictor:
    def __init__(self, grado=4):
        self.grado = grado
        self.model = None
        self.poly = None

    def entrenar_modelo(self, ratings):
        """
        Entrena un modelo de regresión polinomial con los ratings históricos.
        """
        x = np.arange(1, len(ratings) + 1).reshape(-1, 1)
        y = np.array(ratings)

        self.poly = PolynomialFeatures(degree=self.grado)
        x_poly = self.poly.fit_transform(x)

        self.model = LinearRegression()
        self.model.fit(x_poly, y)

    def predecir_ratings(self, ratings, partidos_a_predecir=5):
        """
        Predice los ratings para los próximos partidos.
        """
        if self.model is None or self.poly is None:
            raise ValueError("El modelo no ha sido entrenado.")

        x_pred = np.arange(len(ratings) + 1, len(ratings) + 1 + partidos_a_predecir).reshape(-1, 1)
        x_poly_pred = self.poly.transform(x_pred)
        predicciones = self.model.predict(x_poly_pred)

        print(f"Predicciones de ratings para los próximos {partidos_a_predecir} partidos: {predicciones}")

        return predicciones

    def calcular_precision(self, predicciones, valores_reales):
        """
        Compara las predicciones con los valores reales e imprime métricas de error.
        """
        if len(predicciones) != len(valores_reales):
            print("Advertencia: No hay suficientes valores reales para evaluar todas las predicciones.")
            return

        # Calcular error absoluto medio (MAE)
        mae = mean_absolute_error(valores_reales, predicciones)
        print(f"\nError Absoluto Medio (MAE): {mae:.4f}")

        # Calcular precisión relativa
        precision_relativa = 100 - (mae / np.mean(valores_reales)) * 100
        print(f"Precisión relativa promedio: {precision_relativa:.2f}%")
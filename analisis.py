import ScraperFC as sfc
from integracion import DataExtractor, DataSaver, RatingPredictor
import time

def main():
    # Registrar el tiempo inicial
    inicio = time.time()
    print("Ejecutando el script...")

    # Crear una instancia del scraper de SofaScore
    scraper = sfc.Sofascore()

    # Definir la liga y la temporada
    league = 'EPL'  # Premier League
    season = '23/24'  # Temporada 2023/2024

    # Lista de jugadores a extraer
    jugadores = ['Bryan Mbeumo', 'João Pedro']

    # Instanciar las clases de extracción y guardado
    extractor = DataExtractor(scraper)
    saver = DataSaver()
    predictor = RatingPredictor()

    estadisticas_jugadores = {}

    for jugador in jugadores:
        print(f"Obteniendo estadísticas para {jugador}...")
        estadisticas = extractor.obtener_estadisticas_jugador_optimizado(jugador, league, season)

        if not estadisticas.empty:
            estadisticas_jugadores[jugador] = estadisticas

    # Guardar las estadísticas en CSV
    saver.guardar_estadisticas_csv(estadisticas_jugadores, 'estadisticas_jugadores.csv')

    # Realizar predicción de ratings para cada jugador
    for jugador, df in estadisticas_jugadores.items():
        print(f"\nEntrenando modelo y prediciendo ratings para {jugador}...")
        
        # Extraer ratings del dataframe
        ratings = df['rating_partido'].dropna().tolist()

        if len(ratings) >= 5:
            predictor.entrenar_modelo(ratings)
            predicciones = predictor.predecir_ratings(ratings)

            # Solicitar valores reales al usuario
            valores_reales = []
            print(f"\nIngrese los valores reales de los próximos 5 partidos de {jugador}:")
            for i in range(5):
                while True:
                    try:
                        valor = float(input(f"Partido {i+1}: "))
                        valores_reales.append(valor)
                        break
                    except ValueError:
                        print("Entrada inválida. Introduzca un número.")

            # Evaluar precisión con los valores ingresados
            predictor.calcular_precision(predicciones, valores_reales)

        else:
            print(f"No hay suficientes datos para predecir el rating de {jugador}.")

    # Registrar el tiempo final
    fin = time.time()
    print(f"\nEl tiempo de ejecución del script fue de {fin - inicio:.2f} segundos.")

if __name__ == "__main__":
    main()

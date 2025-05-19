from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar modelo entrenado
modelo = joblib.load("modelo_huracanes.pkl")

# Crear instancia de la app
app = FastAPI(title="API de Predicción de Huracanes")

# Clase para entrada de datos
class DatosHuracan(BaseModel):
    SEASON: int
    DIST2LAND: float
    LANDFALL: float
    USA_LAT: float
    USA_LON: float
    USA_R34_NE: float
    USA_R34_SE: float
    USA_R34_SW: float
    USA_R34_NW: float
    USA_R50_NE: float
    USA_R50_SE: float
    USA_R50_SW: float
    USA_R50_NW: float
    USA_R64_NE: float
    USA_R64_SE: float
    USA_R64_SW: float
    USA_R64_NW: float
    year: int
    month: int
    day: int
    hour: int
    minute: int
    # más variables 

@app.post("/predict")
def predecir_intensidad(datos: DatosHuracan):
    # Convertir entrada a array en el orden correcto
    input_data = np.array([[
        datos.SEASON,
        datos.DIST2LAND,
        datos.LANDFALL,
        datos.USA_LAT,
        datos.USA_LON,
        datos.USA_R34_NE, datos.USA_R34_SE, datos.USA_R34_SW, datos.USA_R34_NW,
        datos.USA_R50_NE, datos.USA_R50_SE, datos.USA_R50_SW, datos.USA_R50_NW,
        datos.USA_R64_NE, datos.USA_R64_SE, datos.USA_R64_SW, datos.USA_R64_NW,
        datos.year, datos.month, datos.day, datos.hour, datos.minute
        # agregar más columnas 
    ]])

    # Hacer la predicción
    prediccion = modelo.predict(input_data)

    return {
        "prediccion_velocidad_viento": round(float(prediccion[0]), 2)
    }

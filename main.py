from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
import uvicorn

# Cargar modelo entrenado
modelo = joblib.load("model/modelo_huracanes.pkl")

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
    BASIN: str
    SUBBASIN: str
    USA_RECORD: str
    USA_STATUS: str
    NATURE: str
    year: int
    month: int
    day: int
    hour: int
    minute: int


@app.post("/predict")
def predecir_intensidad(datos: DatosHuracan):
    try:
        # Listas con las posibles categorías que espera el modelo
        BASINS = ['NI', 'SI', 'SP', 'WP', 'nan']
        SUBBASINS = ['BB', 'CP', 'CS', 'EA', 'GM', 'MM', 'WA', 'nan']
        USA_RECORDS = ['I', 'L', 'R']
        USA_STATUSES = ['HU', 'ST', 'TS', 'TY']
        NATURES = ['MX', 'TS']

        # Generar diccionario base con las variables numéricas
        input_dict = {
            "SEASON": datos.SEASON,
            "DIST2LAND": datos.DIST2LAND,
            "LANDFALL": datos.LANDFALL,
            "USA_LAT": datos.USA_LAT,
            "USA_LON": datos.USA_LON,
            "USA_R34_NE": datos.USA_R34_NE,
            "USA_R34_SE": datos.USA_R34_SE,
            "USA_R34_SW": datos.USA_R34_SW,
            "USA_R34_NW": datos.USA_R34_NW,
            "USA_R50_NE": datos.USA_R50_NE,
            "USA_R50_SE": datos.USA_R50_SE,
            "USA_R50_SW": datos.USA_R50_SW,
            "USA_R50_NW": datos.USA_R50_NW,
            "USA_R64_NE": datos.USA_R64_NE,
            "USA_R64_SE": datos.USA_R64_SE,
            "USA_R64_SW": datos.USA_R64_SW,
            "USA_R64_NW": datos.USA_R64_NW,
            "year": datos.year,
            "month": datos.month,
            "day": datos.day,
            "hour": datos.hour,
            "minute": datos.minute
        }

        # Agregar dummies
        for val in BASINS:
            input_dict[f"BASIN_{val}"] = 1 if datos.BASIN == val else 0
        for val in SUBBASINS:
            input_dict[f"SUBBASIN_{val}"] = 1 if datos.SUBBASIN == val else 0
        for val in USA_RECORDS:
            input_dict[f"USA_RECORD_{val}"] = 1 if datos.USA_RECORD == val else 0
        for val in USA_STATUSES:
            input_dict[f"USA_STATUS_{val}"] = 1 if datos.USA_STATUS == val else 0
        for val in NATURES:
            input_dict[f"NATURE_{val}"] = 1 if datos.NATURE == val else 0

        # Convertir a DataFrame ordenado según las columnas del modelo
        input_ordered = pd.DataFrame([input_dict], columns=modelo.feature_names_in_)

        # Hacer predicción
        prediccion = modelo.predict(input_ordered)

        return {
            "prediccion_velocidad_viento": round(float(prediccion[0]), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la predicción: {str(e)}")



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
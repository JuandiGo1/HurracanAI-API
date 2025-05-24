# -*- coding: utf-8 -*-
"""
PF_Modelo_Huracanes.py

Adaptado de un notebook de Colab.
Se enfoca en la preparación de datos y el entrenamiento del modelo de regresión.
"""

# !pip install umap-learn # Comentado, ya que la parte de UMAP/DBSCAN (y su gráfica) se ha eliminado.
                         # Si necesitas umap para otra cosa, descomenta y asegúrate de tenerlo.

import pandas as pd
import numpy as np
# import umap # Eliminado ya que no se usa para el modelo de regresión
# from sklearn.cluster import DBSCAN # Eliminado ya que no se usa para el modelo de regresión
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


"""# Codigo

## Carga del dataset
"""

# Intenta cargar desde un archivo local. Ajusta la ruta si es necesario.
try:
    df = pd.read_csv('ibtracsLast3years.csv')
except FileNotFoundError:
    print("Error: 'ibtracsLast3years.csv' no encontrado. Asegúrate de que el archivo esté en el directorio correcto.")
    exit()

if not df.empty:
    df.drop(df.index[0], inplace=True) #para que no aparezca la unidad de medida
else:
    print("Error: El DataFrame está vacío después de intentar cargar el CSV.")
    exit()

print("Información inicial del DataFrame:")
df.info() #todo desde el inicio viene con tipos mixtos y mayoritariamente como object, hay que cambiar eso

y_target_col = ['USA_WIND'] #target

#inputs
xy_features = ['USA_WIND', 'SEASON', 'BASIN', 'SUBBASIN', 'ISO_TIME', 'NATURE', 'DIST2LAND', "LANDFALL",'USA_LAT', 'USA_LON', 'USA_RECORD',
     'USA_STATUS', 'USA_PRES', 'USA_SSHS', 'USA_R34_NE', 'USA_R34_SE',
     'USA_R34_SW', 'USA_R34_NW', 'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW',
     'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW']

df_ = df[xy_features].copy() # df_ se va a usar para la limpieza y transformaciones

"""## EDA y transformaciones"""

print("\nRevisión de valores únicos por columna (antes de la limpieza):")
columnas_originales = list(df_.keys())
for col_name in columnas_originales:
  print(f"Columna: {col_name}")
  # Para evitar imprimir demasiados valores únicos, limitamos la muestra
  unique_values = df_[col_name].unique()
  if len(unique_values) > 20:
      print(f"  Primeros 20 valores únicos: {unique_values[:20]}")
  else:
      print(f"  Valores únicos: {unique_values}")
  print("---------")

print("\nInformación del DataFrame antes de la corrección de tipos:")
df_.info()

df_ = df_.astype({
    'SEASON': 'int',
    'BASIN': 'str',
    'SUBBASIN': 'str',
    'USA_RECORD': 'str',
    'USA_STATUS': 'str',
    'NATURE': 'str'
})

columns_to_fix_numeric = ['USA_LAT', 'USA_LON', 'USA_PRES', 'USA_SSHS', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW',
                          'USA_R34_NW', 'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW', 'USA_R64_NE',
                          'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW', 'USA_WIND', 'DIST2LAND', 'LANDFALL']

print("\nProcesando columnas numéricas para convertir y manejar NaNs...")
for col in columns_to_fix_numeric:
    # Reemplazar valores no numéricos o marcadores de nulos por np.nan
    df_[col] = df_[col].replace([' ', '-99', -99, '               NaN'], np.nan) # Añadido '               NaN' si es un string literal
    # Convertir a numérico, los errores se convertirán en NaT/NaN
    df_[col] = pd.to_numeric(df_[col], errors='coerce')


# Convertir ISO_TIME a datetime
df_['ISO_TIME'] = pd.to_datetime(df_['ISO_TIME'], errors='coerce')

# Eliminar filas con cualquier valor nulo después de las conversiones
df_.dropna(inplace=True)

print("\nInformación del DataFrame después de la limpieza y conversión de tipos:")
df_.info()

# Definición de columnas numéricas y categóricas (usadas para describe)
col_num = ['USA_WIND','SEASON','DIST2LAND','LANDFALL',
           'USA_LAT','USA_LON','USA_PRES','USA_SSHS',
           'USA_R34_NE','USA_R34_SE','USA_R34_SW','USA_R34_NW',
           'USA_R50_NE','USA_R50_SE','USA_R50_SW','USA_R50_NW',
           'USA_R64_NE','USA_R64_SE','USA_R64_SW','USA_R64_NW']

col_cat = ['BASIN', 'SUBBASIN', 'USA_RECORD', 'USA_STATUS', 'NATURE']


print(df_.isnull().sum())

# Creación del DataFrame para el modelo
df_model = df_.copy()

# Creación de variables dummy para las categóricas
print("\nCreando variables dummy...")

dummies = pd.get_dummies(df_model[col_cat], drop_first=True)
df_model.drop(columns=col_cat, axis=1, inplace=True)
df_model = pd.concat([df_model, dummies], axis=1)



# Extracción de características de fecha y hora

df_model['year'] = df_model['ISO_TIME'].dt.year
df_model['month'] = df_model['ISO_TIME'].dt.month
df_model['day'] = df_model['ISO_TIME'].dt.day
df_model['hour'] = df_model['ISO_TIME'].dt.hour
df_model['minute'] = df_model['ISO_TIME'].dt.minute
df_model = df_model.drop(columns=['ISO_TIME'])



print("\nInformación del DataFrame listo para el modelo:")
df_model.info()
print("\nPrimeras filas del DataFrame para el modelo:")
print(df_model.head())

"""## Entrenamiento del modelo"""

target_variable = 'USA_WIND'

if target_variable not in df_model.columns:
    print(f"Error: La variable objetivo '{target_variable}' no se encuentra en df_model.")
    exit()

X = df_model.drop(columns=[target_variable])

# Columnas a eliminar de las características (features)
cols_to_drop_from_X = []
if 'USA_SSHS' in X.columns:
    cols_to_drop_from_X.append('USA_SSHS')
    print("Eliminando 'USA_SSHS' de las características (explicativa de la variable objetivo).")
if 'USA_PRES' in X.columns:
    cols_to_drop_from_X.append('USA_PRES')
    print("Eliminando 'USA_PRES' de las características (alta correlación inversa con velocidad del viento).")
if 'SEASON' in X.columns:
    cols_to_drop_from_X.append('SEASON')
    print("Eliminando 'SEASON' de las características (no debe usarse como feature).")
if 'year' in X.columns:
    cols_to_drop_from_X.append('year')
    print("Eliminando 'year' de las características (no debe usarse como feature).")

if cols_to_drop_from_X:
    X = X.drop(columns=cols_to_drop_from_X)

Y = df_model[target_variable]

if X.empty:
    print("Error: El DataFrame de características X está vacío. Revise los pasos de preprocesamiento.")
    exit()

print(f"\nForma de X: {X.shape}, Forma de Y: {Y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(f"Tamaño de X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Tamaño de y_train: {y_train.shape}, y_test: {y_test.shape}")


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  # Usa todos los procesadores disponibles
)

print("\nEntrenando el modelo RandomForestRegressor...")
model.fit(X_train, y_train)
print("Modelo entrenado.")

# Importancia de cada variable dentro del modelo
print("\nImportancia de las características en el modelo:")
importances = model.feature_importances_
features = X_train.columns

importancia_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(importancia_df.head(15))  # Top 15 variables más importantes

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)

# R² (Coeficiente de determinación)
r2 = r2_score(y_test, y_pred)

# Mostrar resultados
print("\nResultados de la evaluación del modelo:")
print(f'RMSE: {rmse:.2f} nudos')
print(f'MAE: {mae:.2f} nudos')
print(f'R²: {r2:.3f}')

print("\n--- Proceso completado ---")

# Guardar el modelo entrenado en un archivo .pkl
joblib.dump(model, 'modelo_huracanes.pkl')
print("Modelo exportado como 'modelo_huracanes.pkl'")

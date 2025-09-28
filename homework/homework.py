# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
import os
import gzip
import pickle
import json
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix



def cargar_dataset(ruta: str) -> pd.DataFrame:
    
    return pd.read_csv(ruta, index_col=False, compression="zip")


def limpiar_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    # 1. Renombrar la columna objetivo para que sea más simple.
    dataframe = dataframe.rename(columns={"default payment next month": "default"})
    
    # 2. Eliminar la columna "ID" ya que no aporta valor predictivo.
    dataframe = dataframe.drop(columns=["ID"])
    
    # 3. Eliminar registros con valores no disponibles (0) en MARRIAGE y EDUCATION.
    dataframe = dataframe.loc[dataframe["MARRIAGE"] != 0]
    dataframe = dataframe.loc[dataframe["EDUCATION"] != 0]
    
    # 4. Agrupar niveles de educación > 3 en la categoría 4 ("others").
    dataframe["EDUCATION"] = dataframe["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    
    return dataframe


def crear_pipeline() -> Pipeline:
    """
    Crea el pipeline de preprocesamiento y el clasificador.

    Returns:
        Pipeline: Un objeto Pipeline de scikit-learn listo para ser entrenado.
    """
    # Se definen las columnas que serán tratadas como categóricas.
    # ¡Importante! Las columnas PAY_* son categóricas (estado de pago), no numéricas.
    # Tratarlas como categóricas es crucial para el rendimiento del modelo.
    columnas_categoricas = [
        "SEX", "EDUCATION", "MARRIAGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
    ]
    
    # El preprocesador aplica OneHotEncoding a las características categóricas
    # y deja el resto de columnas (numéricas) sin cambios ("passthrough").
    preprocesador = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)],
        remainder="passthrough",
    )
    
    # El pipeline final une el preprocesamiento con el clasificador RandomForest.
    # Se fija el random_state para que los resultados sean reproducibles.
    flujo_de_trabajo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("clasificador", RandomForestClassifier(random_state=42)),
        ]
    )
    
    return flujo_de_trabajo


def crear_estimador(pipeline: Pipeline) -> GridSearchCV:
    """
    Crea el objeto GridSearchCV para la optimización de hiperparámetros.

    Args:
        pipeline (Pipeline): El pipeline que se va a optimizar.

    Returns:
        GridSearchCV: El objeto GridSearchCV configurado.
    """
    # Se define una "grilla" con los hiperparámetros que se quieren probar.
    # GridSearchCV probará todas las combinaciones posibles.
    grilla_parametros = {
        "clasificador__n_estimators": [200, 300],
        "clasificador__max_depth": [10, 20, 30],
        "clasificador__min_samples_split": [5, 10],
        "clasificador__min_samples_leaf": [2, 4],
    }

    # Se configura GridSearchCV:
    # - cv=10: Usará validación cruzada de 10 particiones (folds).
    # - scoring='balanced_accuracy': Medirá el rendimiento con la precisión balanceada.
    # - n_jobs=-1: Usará todos los núcleos de CPU disponibles para acelerar la búsqueda.
    estimador = GridSearchCV(
        pipeline,
        grilla_parametros,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
        refit=True,  # Re-entrena el mejor modelo con todos los datos al final.
    )
    
    return estimador


def guardar_modelo(estimador: GridSearchCV, ruta: str):
    """
    Guarda el modelo entrenado en formato pkl comprimido con gzip.

    Args:
        estimador (GridSearchCV): El modelo (estimador) ya entrenado.
        ruta (str): La ruta del archivo donde se guardará el modelo.
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with gzip.open(ruta, "wb") as archivo:
        pickle.dump(estimador, archivo)
    print(f"Modelo guardado exitosamente en: {ruta}")


def calcular_metricas_precision(nombre_dataset: str, y_real, y_predicho) -> dict:
    
    return {
        "type": "metrics",
        "dataset": nombre_dataset,
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_predicho),
        "recall": recall_score(y_real, y_predicho, zero_division=0),
        "f1_score": f1_score(y_real, y_predicho, zero_division=0),
    }


def calcular_metricas_confusion(nombre_dataset: str, y_real, y_predicho) -> dict:
    
    matriz_confusion = confusion_matrix(y_real, y_predicho)
    return {
        "type": "cm_matrix",
        "dataset": nombre_dataset,
        "true_0": {"predicted_0": int(matriz_confusion[0][0]), "predicted_1": int(matriz_confusion[0][1])},
        "true_1": {"predicted_0": int(matriz_confusion[1][0]), "predicted_1": int(matriz_confusion[1][1])},
    }


def principal():
    
    ruta_archivos_entrada = "files/input/"
    ruta_archivos_modelos = "files/models/"
    ruta_archivos_salida = "files/output/"

    print("Cargando datasets...")
    df_prueba = cargar_dataset(os.path.join(ruta_archivos_entrada, "test_data.csv.zip"))
    df_entrenamiento = cargar_dataset(os.path.join(ruta_archivos_entrada, "train_data.csv.zip"))

    print("Limpiando datasets...")
    df_prueba_limpio = limpiar_dataset(df_prueba)
    df_entrenamiento_limpio = limpiar_dataset(df_entrenamiento)

    # --- 4. División en Características (X) y Objetivo (y) ---
    x_prueba = df_prueba_limpio.drop(columns=["default"])
    y_prueba = df_prueba_limpio["default"]
    x_entrenamiento = df_entrenamiento_limpio.drop(columns=["default"])
    y_entrenamiento = df_entrenamiento_limpio["default"]

    # --- 5. Creación y Entrenamiento del Modelo ---
    print("Creando el pipeline...")
    pipeline = crear_pipeline()
    
    print("Creando el estimador con GridSearchCV...")
    estimador_final = crear_estimador(pipeline)
    
    print("Iniciando el entrenamiento (esto puede tardar varios minutos)...")
    estimador_final.fit(x_entrenamiento, y_entrenamiento)
    print("Entrenamiento finalizado.")
    print(f"Mejor score (balanced_accuracy) en CV: {estimador_final.best_score_:.4f}")
    print(f"Mejores parámetros encontrados: {estimador_final.best_params_}")

    # --- 6. Guardado del Modelo ---
    guardar_modelo(estimador_final, os.path.join(ruta_archivos_modelos, "model.pkl.gz"))

    # --- 7. Predicción y Evaluación ---
    print("Realizando predicciones y calculando métricas...")
    y_prueba_predicho = estimador_final.predict(x_prueba)
    y_entrenamiento_predicho = estimador_final.predict(x_entrenamiento)

    metricas_precision_entrenamiento = calcular_metricas_precision("train", y_entrenamiento, y_entrenamiento_predicho)
    metricas_precision_prueba = calcular_metricas_precision("test", y_prueba, y_prueba_predicho)

    metricas_confusion_entrenamiento = calcular_metricas_confusion("train", y_entrenamiento, y_entrenamiento_predicho)
    metricas_confusion_prueba = calcular_metricas_confusion("test", y_prueba, y_prueba_predicho)

    # --- 8. Guardado de Métricas ---
    os.makedirs(ruta_archivos_salida, exist_ok=True)
    with open(os.path.join(ruta_archivos_salida, "metrics.json"), "w") as archivo:
        archivo.write(json.dumps(metricas_precision_entrenamiento) + "\n")
        archivo.write(json.dumps(metricas_precision_prueba) + "\n")
        archivo.write(json.dumps(metricas_confusion_entrenamiento) + "\n")
        archivo.write(json.dumps(metricas_confusion_prueba) + "\n")
    
    print(f"Métricas guardadas exitosamente en: {ruta_archivos_salida}metrics.json")

if __name__ == "__main__":
    principal()
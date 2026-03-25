from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

def entrenar_modelo():
    # Ruta correcta del CSV (esto evita el error FileNotFound)
    ruta = os.path.join(os.path.dirname(__file__), "dataset_regresion_logistica.csv")
    df = pd.read_csv(ruta)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return modelo, acc


@app.route("/", methods=["GET", "POST"])
def index():
    modelo, acc = entrenar_modelo()
    prediccion = None

    if request.method == "POST":
        try:
            edad = int(request.form["edad"])
            ingreso = float(request.form["ingreso"])
            visitas = int(request.form["visitas"])
            tiempo = float(request.form["tiempo"])
            compras = int(request.form["compras"])
            descuento = int(request.form["descuento"])

            datos = [[edad, ingreso, visitas, tiempo, compras, descuento]]
            resultado = modelo.predict(datos)

            prediccion = "Compra" if resultado[0] == 1 else "No compra"
        except:
            prediccion = "Error en los datos ingresados"

    return render_template("index.html", accuracy=acc, prediccion=prediccion)


if __name__ == "__main__":
    app.run(debug=True)
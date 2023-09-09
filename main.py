from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
# Cargar el modelo entrenado
model = joblib.load('models/modelo_regresion.pkl')  # Cargar el modelo previamente guardado

# Crear una aplicación Flask
app = Flask(__name__)

# Definir la ruta principal del sitio web
@app.route('/')
def index():
    return render_template('index.html')  # Renderizar la plantilla 'index.html'

# Definir la ruta para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los valores del formulario enviado
    nitrogeno = float(request.form['nitrogeno'])
    fosforo = float(request.form['fosforo'])
    potasio = float(request.form['potasio'])
    Temperatura = float(request.form['temperatura'])
    Humedad = float(request.form['humedad'])
    ph = float(request.form['ph'])
    Precipitacion = float(request.form['precipitacion'])
    scaler = StandardScaler()
    
    # Valores futuros
    new_samples = np.array([[nitrogeno, fosforo, potasio, Temperatura, Humedad, ph, Precipitacion]])
    
    prediccion = model.predict(new_samples)
    # Iniciar la aplicación si este script es el punto de entrada
    return render_template('result.html', pred=prediccion[0]) 

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))

import numpy as np
import pandas as pd
import math
from jinja2 import Template
import webbrowser
import os
#python app.py para ejecutarlo en terminal  
# Datos originales se modifica todo depende de los valores que añadira
data = {
    'Paciente': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7','P8','P9','P10','P11','P12'],
    'x': [9, 6, 8, 6, 5, 7, 6,7,5,5,4,5],
    'y_original': [0, 0, 0,0, 1, 0,0,0,1, 1, 1, 1]
}

# Crear DataFrame
df = pd.DataFrame(data)

# PASO 1: Ajuste de valores de y
df['y_ajustado'] = df['y_original'].replace({0: 0.01, 1: 0.99})

# PASO 2: Cálculo de logit(y)
df['logit_y'] = np.log(df['y_ajustado'] / (1 - df['y_ajustado']))

# PASO 3: Cálculos para la regresión
df['x_logit_y'] = df['x'] * df['logit_y']
df['x_cuadrado'] = df['x'] ** 2

# Sumatorias
n = len(df)
sum_x = df['x'].sum()
sum_logit_y = df['logit_y'].sum()
sum_x_logit_y = df['x_logit_y'].sum()
sum_x_cuadrado = df['x_cuadrado'].sum()

# Cálculo de coeficientes
b1 = (n * sum_x_logit_y - sum_x * sum_logit_y) / (n * sum_x_cuadrado - sum_x ** 2)
b0 = (sum_logit_y - b1 * sum_x) / n

# Función para predecir probabilidades
def predict_probability(x):
    logit = b0 + b1 * x
    probability = 1 / (1 + math.exp(-logit))
    return probability

# Generar predicciones
predictions = [{'x': x, 'prob': predict_probability(x), 
                'class': 'Aprobado' if predict_probability(x) >= 0.5 else 'Reprobado'} 
               for x in range(1, 8)]

# Plantilla HTML
html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Regresión Logística Simple</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1, h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 3px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e6f7ff;
        }
        .formula {
            background-color: #f0f8ff;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
        }
        .result-box {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #2ecc71;
        }
        .prediction {
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            border-radius: 20px;
            background-color: #e8f8f5;
            color: #27ae60;
            font-weight: bold;
        }
        .prediction.reprobado {
            background-color: #fae8e8;
            color: #e74c3c;
        }
    </style>
</head>
<body>
    <h1>Regresión Logística Simple</h1>
    
    <h2>Datos Originales</h2>
    <table>
        <tr>
            <th>Paciente</th>
            <th>Horas de Paciente (x)</th>
            <th>Aprobó (y)</th>
        </tr>
        {% for row in data %}
        <tr>
            <td>{{ row.Paciente }}</td>
            <td>{{ row.x }}</td>
            <td>{{ row.y_original }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Paso 1: Ajuste de valores de y</h2>
    <div class="result-box">
        <p>Para evitar valores extremos al aplicar el logaritmo:</p>
        <p><strong>0 → 0.01</strong> y <strong>1 → 0.99</strong></p>
    </div>
    
    <h2>Paso 2: Cálculo de logit(y)</h2>
    <div class="formula">
        logit(y) = ln(y / (1 - y))
    </div>
    <table>
        <tr>
            <th>Paciente</th>
            <th>y (ajustado)</th>
            <th>logit(y)</th>
        </tr>
        {% for row in data %}
        <tr>
            <td>{{ row.Paciente }}</td>
            <td>{{ row.y_ajustado }}</td>
            <td>{{ row.logit_y|round(3) }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Paso 3: Tabla de operaciones</h2>
    <table>
        <tr>
            <th>x</th>
            <th>y (ajustado)</th>
            <th>logit(y)</th>
            <th>x·logit(y)</th>
            <th>x²</th>
        </tr>
        {% for row in data %}
        <tr>
            <td>{{ row.x }}</td>
            <td>{{ row.y_ajustado }}</td>
            <td>{{ row.logit_y|round(3) }}</td>
            <td>{{ row.x_logit_y|round(3) }}</td>
            <td>{{ row.x_cuadrado }}</td>
        </tr>
        {% endfor %}
        <tr style="font-weight: bold; background-color: #e6f7ff;">
            <td>∑ = {{ sum_x }}</td>
            <td></td>
            <td>∑ = {{ sum_logit_y|round(3) }}</td>
            <td>∑ = {{ sum_x_logit_y|round(2) }}</td>
            <td>∑ = {{ sum_x_cuadrado }}</td>
        </tr>
    </table>
    
    <h2>Paso 4: Cálculo de la pendiente (b₁)</h2>
    <div class="formula">
        b₁ = [n·∑(x·logit(y)) - ∑x·∑logit(y)] / [n·∑x² - (∑x)²]<br>
        b₁ = [{{ n }}·{{ sum_x_logit_y|round(2) }} - {{ sum_x }}·{{ sum_logit_y|round(3) }}] / [{{ n }}·{{ sum_x_cuadrado }} - {{ sum_x }}²]<br>
        b₁ = {{ b1|round(2) }}
    </div>
    
    <h2>Paso 5: Cálculo del intercepto (b₀)</h2>
    <div class="formula">
        b₀ = [∑logit(y) - b₁·∑x] / n<br>
        b₀ = [{{ sum_logit_y|round(3) }} - {{ b1|round(2) }}·{{ sum_x }}] / {{ n }}<br>
        b₀ = {{ b0|round(2) }}
    </div>
    
    <h2>Paso 6: Ecuación Logística Final</h2>
    <div class="result-box">
        <div class="formula">
            logit(p) = {{ b0|round(2) }} + {{ b1|round(2) }}·x<br><br>
            p(x) = 1 / (1 + e<sup>-({{ b0|round(2) }} + {{ b1|round(2) }}·x)</sup>)
        </div>
    </div>
    
    <h2>Predicciones</h2>
    <div style="margin-bottom: 20px;">
        {% for pred in predictions %}
        <div class="prediction {% if pred.class == 'Reprobado' %}reprobado{% endif %}">
            x = {{ pred.x }}: p(x) = {{ pred.prob|round(4) }} → {{ pred.class }}
        </div>
        {% endfor %}
    </div>
    
    <div style="text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 0.9em;">
        Reporte generado automáticamente con Python
    </div>
</body>
</html>
"""

# Preparar datos para la plantilla
template_data = {
    'data': df.to_dict('records'),
    'sum_x': sum_x,
    'sum_logit_y': sum_logit_y,
    'sum_x_logit_y': sum_x_logit_y,
    'sum_x_cuadrado': sum_x_cuadrado,
    'n': n,
    'b1': b1,
    'b0': b0,
    'predictions': predictions
}

# Renderizar HTML
template = Template(html_template)
html_output = template.render(template_data)

# Guardar y abrir el archivo HTML
output_file = "regresion_logistica.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_output)

# Abrir en el navegador
webbrowser.open('file://' + os.path.realpath(output_file))
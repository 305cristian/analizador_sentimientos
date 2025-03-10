from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import pandas as pd
from analizador.sentiment_analyzer import analyze_sentiment, create_pie_chart, create_wordclouds

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs('static/images', exist_ok=True)

# Variable global para almacenar el DataFrame analizado
df_global = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#Ruta raiz
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para cargar los archivos
@app.route('/upload', methods=['POST'])
def upload_file():
    global df_global
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Carga el CSV
            df = pd.read_csv(file_path)
            
            # Analiza sentimientos
            df_global, error = analyze_sentiment(df)
            
            if error:
                flash(error)
                return redirect(url_for('index'))
            
            # Creación de graficos (pastel y nubes de palabras)
            sentiment_counts = create_pie_chart(df_global)
            wordcloud_paths = create_wordclouds(df_global)
            
            # Datos para la tabla (se limita a 100 filas para la vista)
            table_data = df_global[['text', 'compound', 'result']].head(100).to_dict('records')
            
            return render_template('results.html', 
                                  sentiment_counts=sentiment_counts,
                                  wordcloud_paths=wordcloud_paths,
                                  table_data=table_data,
                                  filename=filename)
        
        except Exception as e:
            flash(f'Error al procesar el archivo: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Tipo de archivo no permitido. Solo se permiten archivos CSV.')
    return redirect(url_for('index'))

# Ruta de descarga de informe
@app.route('/download-results')
def download_results():
    global df_global
    
    if df_global is None:
        flash('No hay datos para descargar')
        return redirect(url_for('index'))
    
    # Se crea un nombre de archivo temporal para almacenar los resultados en un csv
    output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment_results.csv')
    df_global.to_csv(output_filename, index=False)
    
    return redirect(url_for('static', filename='uploads/sentiment_results.csv'))

if __name__ == '__main__':
    app.run(debug=True)
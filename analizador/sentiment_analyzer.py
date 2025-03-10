
import matplotlib
matplotlib.use('Agg')  # Se Configura matplotlib para entorno sin GUI
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Inicializamos el analizador de sentimientos de VADER
analyzer = SentimentIntensityAnalyzer()

# Función para limpiar texto
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        return text
    return ""

# Función para clasificar el sentimiento
def classify_sentiment(score):
    if score >= 0.05:
        return 1  # Si es Positivo
    elif score <= -0.05:
        return -1  # Si es Negativo
    else:
        return 0  # Si es Neutro

# Función para analizar sentimiento del DataFrame
def analyze_sentiment(df):
    # Asegurarnos que existe la columna 'text'
    if 'text' not in df.columns:
        if len(df.columns) > 0:  #La primera columna debe contener el texto
            df['text'] = df[df.columns[0]]
        else:
            return None, "El archivo no tiene columnas"
    
    # Limpiamos el texto
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Aplicamos el análisis de sentimiento usando VADER
    df['vader_scores'] = df['clean_text'].apply(lambda text: analyzer.polarity_scores(text))
    
    # Extraemos el compound score (puntaje compuesto)
    df['compound'] = df['vader_scores'].apply(lambda score_dict: score_dict['compound'])
    
    # Clasificamos el sentimiento basado en el compound score
    df['result'] = df['compound'].apply(classify_sentiment)
    
    return df, None

# Función para crear gráfica de pastel
def create_pie_chart(df_global):

    sentiment_counts = df_global['result'].value_counts().sort_index()
    
    # Creamos el gráfico de pastel
    labels = ['Negativo', 'Neutro', 'Positivo']
    colors = ['#e15759', '#f28e2b', '#4e79a7']  # Rojo, Amarillo, Azul
    
    # Creamos valores para el gráfico (en el orden correcto)
    values = [sentiment_counts.get(-1, 0), sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]
    
    plt.figure(figsize=(10, 6))
    plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Análisis de Sentimiento con VADER', fontsize=18)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('static/images/sentiment_pie_chart.png')
    plt.close()
    
    return {
        'negative': int(sentiment_counts.get(-1, 0)),
        'neutral': int(sentiment_counts.get(0, 0)),
        'positive': int(sentiment_counts.get(1, 0)),
    }

# Función para crear nubes de palabras
def create_wordclouds(df_global):
    sentiment_types = [
        {'value': -1, 'title': 'Palabras en Comentarios Negativos', 'colormap': 'Reds', 'filename': 'wordcloud_neg.png'},
        {'value': 0, 'title': 'Palabras en Comentarios Neutros', 'colormap': 'YlOrBr', 'filename': 'wordcloud_neu.png'},
        {'value': 1, 'title': 'Palabras en Comentarios Positivos', 'colormap': 'Blues', 'filename': 'wordcloud_pos.png'}
    ]
    
    wordcloud_paths = []
    
    for sentiment in sentiment_types:
        text = ' '.join(df_global[df_global['result'] == sentiment['value']]['clean_text'])
        if text.strip():  # Verificar que hay texto para este sentimiento
            wordcloud = WordCloud(width=800, height=400, 
                                 colormap=sentiment['colormap'],
                                 background_color='white', 
                                 max_words=100,
                                 contour_width=1, 
                                 contour_color='steelblue').generate(text)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(sentiment['title'], fontsize=18)
            plt.axis('off')
            plt.tight_layout()
            
            file_path = f"static/images/{sentiment['filename']}"
            plt.savefig(file_path)
            plt.close()
            
            wordcloud_paths.append({
                'title': sentiment['title'],
                'path': file_path
            })
    
    return wordcloud_paths
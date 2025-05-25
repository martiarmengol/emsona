from flask import Flask, render_template, request, redirect, url_for
import re
import csv
import os

app = Flask(__name__)

# Función para extraer el ID de video de YouTube de una URL
def obtener_video_id(url):
    if not url:
        return None
    
    # Asegurar que la URL tenga el protocolo para el procesamiento correcto
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        # Método 1: Extraer usando expresiones regulares para varios formatos de URL
        patron = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(patron, url)
        if match:
            return match.group(1)
        
        # Método 2: Extraer de la URL de watch usando parámetros
        if 'youtube.com/watch' in url:
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(url)
            video_id = parse_qs(parsed_url.query).get('v')
            if video_id and len(video_id[0]) == 11:
                return video_id[0]
        
        # Método 3: Extraer de URL corta youtu.be
        if 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0].split('&')[0]
            if len(video_id) == 11:
                return video_id
        
        return None
    except Exception as e:
        print(f"Error al extraer ID de video: {e}")
        return None

# Función para leer el archivo CSV de canciones
def leer_canciones_csv():
    canciones = []
    try:
        csv_path = os.path.join(app.static_folder, 'csv', 'songs.csv')
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                canciones.append(row)
        return canciones
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html', pagina_activa='inicio')

@app.route('/loading', methods=['POST'])
def loading():
    youtube_url = request.form.get('youtube_url')
    video_id = obtener_video_id(youtube_url)
    print(f"URL recibida: {youtube_url}")
    print(f"ID extraído: {video_id}")
    # Guardar la URL y el ID en la sesión para usarlos en la página de resultados
    return render_template('loading.html', youtube_url=youtube_url, video_id=video_id, pagina_activa='inicio')

@app.route('/results')
def results():
    # En una aplicación real, obtendrías el video_id de la sesión o de un parámetro en la URL
    video_id = request.args.get('video_id', 'Z5LVw2abUlw')
    print(f"ID recibido en resultados: {video_id}")
    
    # Aquí normalmente procesarías la URL y obtendrías recomendaciones
    # Por ahora, usaremos datos de ejemplo
    canciones_similares = [
        {"titulo": "El Cant dels Ocells", "artista": "Pau Casals", "similitud": "95%", "link": "https://www.youtube.com/embed/Z5LVw2abUlw", "id": "Z5LVw2abUlw"},
        {"titulo": "L'Empordà", "artista": "Sopa de Cabra", "similitud": "87%", "link": "https://www.youtube.com/embed/Z5LVw2abUlw", "id": "Z5LVw2abUlw"},
        {"titulo": "Boig per tu", "artista": "Sau", "similitud": "82%", "link": "https://www.youtube.com/embed/Z5LVw2abUlw", "id": "Z5LVw2abUlw"},
        {"titulo": "Paraules d'Amor", "artista": "Joan Manuel Serrat", "similitud": "78%", "link": "https://www.youtube.com/embed/Z5LVw2abUlw", "id": "Z5LVw2abUlw"},
        {"titulo": "Bon dia", "artista": "Els Pets", "similitud": "75%", "link": "https://www.youtube.com/embed/Z5LVw2abUlw", "id": "Z5LVw2abUlw"}
    ]
    
    return render_template('results.html', canciones=canciones_similares, video_id_usuario=video_id, pagina_activa='inicio')

@app.route('/library')
def library():
    # Obtener parámetros de búsqueda y paginación
    query = request.args.get('query', '').lower()
    page = int(request.args.get('page', 1))
    per_page = 10
    
    # Leer todas las canciones del CSV
    todas_canciones = leer_canciones_csv()
    
    # Filtrar por búsqueda si hay query
    if query:
        canciones_filtradas = [
            cancion for cancion in todas_canciones 
            if query in cancion['Song Name'].lower() or 
               query in cancion['Band'].lower() or 
               query in cancion['Genre'].lower()
        ]
    else:
        canciones_filtradas = todas_canciones
    
    # Calcular total de páginas
    total_canciones = len(canciones_filtradas)
    total_paginas = (total_canciones + per_page - 1) // per_page
    
    # Obtener canciones para la página actual
    inicio = (page - 1) * per_page
    fin = min(inicio + per_page, total_canciones)
    canciones_pagina = canciones_filtradas[inicio:fin]
    
    return render_template('library.html', 
                          canciones=canciones_pagina,
                          query=query,
                          page=page,
                          total_paginas=total_paginas,
                          total_canciones=total_canciones,
                          pagina_activa='biblioteca')

@app.route('/about')
def about():
    return render_template('about.html', pagina_activa='nosotros')

if __name__ == '__main__':
    app.run(debug=True) 
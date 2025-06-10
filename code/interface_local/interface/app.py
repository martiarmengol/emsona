from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
import csv
import os
import subprocess
import json
import sys
import threading
import uuid
import time
sys.path.append('/home/guillem/Music/interface_local_v4/interface_local')

from recommendator.app.app import recomendator_with_faiss

# Add the full_program directory to sys.path

app = Flask(__name__)

# Diccionario global para almacenar el estado de los procesos
processing_status = {}

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

def is_single_video(url):
    """
    Check if the URL is for a single video or a playlist.
    
    Args:
        url (str): YouTube URL to check
        
    Returns:
        bool: True if it's a single video, False if it's a playlist or invalid URL
    """
    if not url or not isinstance(url, str):
        return False
    
    # Ensure URL has protocol
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    try:
        # Check for playlist indicators (más exhaustivo)
        playlist_indicators = [
            'playlist?',
            '&list=',
            '/playlist/',
            'list=PL',  # Playlists típicamente empiezan con PL
            'list=UU',  # Channel uploads playlist
            'list=FL',  # Favorites playlist
            'list=LL',  # Liked videos playlist
            'list=WL',  # Watch later playlist
        ]
        
        # If any playlist indicator is found, return False
        if any(indicator in url for indicator in playlist_indicators):
            return False
            
        # Check if it's a valid video URL using the existing obtener_video_id function
        video_id = obtener_video_id(url)
        if video_id is None:
            return False
            
        # Additional check: ensure it's not a channel URL
        channel_indicators = [
            '/channel/',
            '/user/',
            '/c/',
            '@'  # New YouTube handle format
        ]
        
        if any(indicator in url for indicator in channel_indicators):
            return False
            
        return True
        
    except Exception as e:
        print(f"Error checking URL type: {e}")
        return False

# Función para procesar la canción en segundo plano
def process_song_background(process_id, youtube_url):
    """Procesa la canción en segundo plano y actualiza el estado"""
    try:
        # Definir las rutas necesarias
        song_path = "./recommendator/app/store_song"
        embedding_path = "./recommendator/app/store_embedding"
        model_path = "./recommendator/app/MODEL/model.pb"
        metadata_path = "./recommendator/app/METADATA/metadata.csv"
        store_metadata_path = "./recommendator/app/store_metadata"
        song_embeddings = "./recommendator/app/EMBEDDINGS/embeddings.pkl"
        store_svg = "./recommendator/app/SVG/plot.svg"
        k = 1
        n = 5

        # ✅ Llamar a la función recomendator_with_faiss (ahora devuelve dict)
        result = recomendator_with_faiss(
            song_path, youtube_url, embedding_path, model_path, metadata_path, store_metadata_path, song_embeddings, store_svg, k, n
        )

        # ✅ Extraer recomendaciones y coordenadas del resultado
        recommendations = result['recommendations']
        coordinates = result['coordinates']

        # Construir la lista de canciones similares
        canciones_similares = []
        for _, rec in recommendations.iterrows():
            # Convert distance to percentage similarity
            distance = float(rec.get('Distance', 0))
            if distance < 10:
                similarity = 100
            elif distance > 30:
                similarity = 0
            else:
                # Linear interpolation between 100% at distance=10 and 0% at distance=30
                similarity = 100 - ((distance - 10) * (100 / 20))
                
            cancion = {
                "titulo": rec.get("Song", "Título desconocido"),
                "artista": rec.get("Artist", "Desconocido"),
                "similitud": f"{similarity:.0f}%",  # Format as percentage without decimals
                "link": rec.get("YT Link", ""),
                "id": rec.get("video_id", "Z5LVw2abUlw")
            }
            canciones_similares.append(cancion)

        # ✅ Actualizar el estado como completado con coordenadas
        processing_status[process_id] = {
            'status': 'completed',
            'result': canciones_similares,
            'coordinates': coordinates,
            'video_id': obtener_video_id(youtube_url)
        }
        
    except Exception as e:
        print(f"Error al procesar la canción: {e}")
        processing_status[process_id] = {
            'status': 'error',
            'error': str(e),
            'video_id': obtener_video_id(youtube_url)
        }

# Función para leer el archivo CSV de canciones
def leer_canciones_csv():
    canciones = []
    try:
        csv_path = os.path.join(app.static_folder, 'csv', 'catalan_music_metadata.csv')
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
    error_type = request.args.get('error')
    error_message = None
    
    if error_type == 'playlist':
        error_message = "Si us plau, envia l'enllaç d'un vídeo únic, no d'una playlist de YouTube."
    elif error_type == 'invalid':
        error_message = "L'enllaç de YouTube no és vàlid. Si us plau, verifica que sigui correcte."
    elif error_type == 'empty':
        error_message = "Si us plau, envia un enllaç de YouTube vàlid."
    
    return render_template('index.html', pagina_activa='inicio', error_message=error_message)


@app.route('/loading', methods=['POST'])
def loading():
    youtube_url = request.form.get('youtube_url')
    
    # ✅ Validaciones de URL
    if not youtube_url or not youtube_url.strip():
        return redirect(url_for('index', error='empty'))
    
    # Verificar que contenga youtube.com o youtu.be
    if 'youtube.com' not in youtube_url and 'youtu.be' not in youtube_url:
        return redirect(url_for('index', error='invalid'))
    
    # Validar que sea un video único, no una playlist
    if not is_single_video(youtube_url):
        return redirect(url_for('index', error='playlist'))
    
    video_id = obtener_video_id(youtube_url)
    
    # Verificar que se pudo extraer el video ID
    if not video_id:
        return redirect(url_for('index', error='invalid'))
    
    # Generar un ID único para este proceso
    process_id = str(uuid.uuid4())
    
    # Inicializar el estado del proceso
    processing_status[process_id] = {
        'status': 'processing',
        'video_id': video_id
    }
    
    # Iniciar el procesamiento en segundo plano
    thread = threading.Thread(target=process_song_background, args=(process_id, youtube_url))
    thread.daemon = True
    thread.start()
    
    print(f"URL recibida: {youtube_url}")
    print(f"ID extraído: {video_id}")
    print(f"Process ID: {process_id}")
    
    # Renderizar la página de loading con el process_id
    return render_template('loading.html', video_id=video_id, process_id=process_id, pagina_activa='inicio')

@app.route('/api/check_status/<process_id>')
def check_status(process_id):
    """API endpoint para verificar el estado del procesamiento"""
    if process_id not in processing_status:
        return jsonify({'status': 'not_found'})
    
    return jsonify(processing_status[process_id])

@app.route('/results')
def results():
    # Obtener el process_id y video_id desde la URL
    process_id = request.args.get('process_id')
    video_id = request.args.get('video_id', 'Z5LVw2abUlw')
    
    print(f"Process ID recibido en resultados: {process_id}")
    print(f"Video ID recibido en resultados: {video_id}")
    
    canciones_similares = []
    coordinates = []
    
    if process_id and process_id in processing_status:
        status_data = processing_status[process_id]
        if status_data['status'] == 'completed':
            canciones_similares = status_data['result']
            coordinates = status_data.get('coordinates', [])
            # Limpiar el estado después de obtener los resultados
            del processing_status[process_id]
        elif status_data['status'] == 'error':
            print(f"Error en el procesamiento: {status_data.get('error', 'Error desconocido')}")
            # Limpiar el estado después del error
            del processing_status[process_id]
    
    return render_template(
        'results.html',
        canciones=canciones_similares,
        coordinates=coordinates,
        video_id_usuario=video_id,
        pagina_activa='inicio'
    )

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
            if query in cancion['song_title'].lower() or 
               query in cancion['channel_name'].lower() or 
               query in cancion['original_title'].lower()
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

@app.route('/liked')
def liked():
    # Datos de ejemplo para las canciones más gustadas
    canciones_gustadas = [
        {"titulo": "L'Empordà", "artista": "Sopa de Cabra", "likes": 1243, "link": "https://www.youtube.com/watch?v=RV2gJPXLslA"},
        {"titulo": "Boig per tu", "artista": "Sau", "likes": 987, "link": "https://www.youtube.com/watch?v=f1h3D9IJDMI"},
        {"titulo": "Bon dia", "artista": "Els Pets", "likes": 845, "link": "https://www.youtube.com/watch?v=mySe9OnYvzM"},
        {"titulo": "Paraules d'Amor", "artista": "Joan Manuel Serrat", "likes": 752, "link": "https://www.youtube.com/watch?v=h9t4_YpZ1RY"},
        {"titulo": "Jo vull ser rei", "artista": "Els Pets", "likes": 621, "link": "https://www.youtube.com/watch?v=lc5kSz6Ga2o"},
        {"titulo": "Qualsevol nit pot sortir el sol", "artista": "Jaume Sisa", "likes": 592, "link": "https://www.youtube.com/watch?v=aaVwzFZXpFo"},
        {"titulo": "El meu avi", "artista": "La Trinca", "likes": 488, "link": "https://www.youtube.com/watch?v=W2u1X9BNeFI"},
        {"titulo": "Fes-te fotre", "artista": "Lax'n'Busto", "likes": 407, "link": "https://www.youtube.com/watch?v=7USdD_NM50U"},
        {"titulo": "Tot és possible", "artista": "Txarango", "likes": 372, "link": "https://www.youtube.com/watch?v=t0U0Iap2omg"},
        {"titulo": "La Flama", "artista": "Obeses", "likes": 327, "link": "https://www.youtube.com/watch?v=zVLGwcKG7iM"}
    ]
    
    return render_template('liked.html', canciones=canciones_gustadas, pagina_activa='liked')

@app.route('/forms', methods=['GET', 'POST'])
def suggest_song():
    error = None

    if request.method == 'POST':
        title  = request.form.get('title', '').strip()
        artist = request.form.get('artist', '').strip()
        genre  = request.form.get('genre', '').strip()

        # simple validation
        if not title or not artist or not genre:
            error = "Por favor completa todos los campos obligatorios."
            return render_template('forms.html', error=error)

        # aquí guardarías la sugerencia en tu base de datos...
        # por ejemplo:
        # db.save({ 'title': title, 'artist': artist, 'album': request.form.get('album'), ... })

        return redirect(url_for('index'))

    # GET → muestro el formulario vacío
    return render_template('forms.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)
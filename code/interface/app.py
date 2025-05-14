from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/loading', methods=['POST'])
def loading():
    youtube_url = request.form.get('youtube_url')
    # Aquí guardarías la URL para procesarla después
    return render_template('loading.html', youtube_url=youtube_url)

@app.route('/results')
def results():
    # Aquí normalmente procesarías la URL y obtendrías recomendaciones
    # Por ahora, usaremos datos de ejemplo
    canciones_similares = [
        {"titulo": "El Cant dels Ocells", "artista": "Pau Casals", "similitud": "95%"},
        {"titulo": "L'Empordà", "artista": "Sopa de Cabra", "similitud": "87%"},
        {"titulo": "Boig per tu", "artista": "Sau", "similitud": "82%"},
        {"titulo": "Paraules d'Amor", "artista": "Joan Manuel Serrat", "similitud": "78%"},
        {"titulo": "Bon dia", "artista": "Els Pets", "similitud": "75%"}
    ]
    return render_template('results.html', canciones=canciones_similares)

if __name__ == '__main__':
    app.run(debug=True) 
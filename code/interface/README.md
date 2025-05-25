# Em Sona - Recomendador de Música Catalana

Aplicación web para descubrir y recomendar música catalana similar a partir de enlaces de YouTube.

## Requisitos

- Python 3.7 o superior
- Flask

## Instalación

1. Instala las dependencias:

```bash
pip install flask
```

## Ejecución

1. Navega al directorio del proyecto:

```bash
cd interface
```

2. Ejecuta la aplicación Flask:

```bash
python app.py
```

3. Abre tu navegador y visita:

```
http://127.0.0.1:5000/
```

## Funcionalidades

- **Página inicial**: Interfaz moderna para ingresar enlaces de YouTube con visualización de ondas de audio
- **Pantalla de carga**: Animación de disco de vinilo y ecualizador durante el análisis
- **Visualización de resultados**: 
  - Mapa t-SNE interactivo que muestra la similitud entre canciones
  - Lista de las 5 canciones más similares con indicadores visuales
  - Interacción entre la lista y el mapa visual
  - Reproducción de videos de YouTube directamente en la interfaz
  - Sistema de valoración de recomendaciones con estrellas (1-5)
- **Biblioteca de canciones**:
  - Listado completo de todas las canciones disponibles en el sistema
  - Buscador por título, artista o género
  - Paginación con 10 canciones por página
  - Enlaces directos a los videos de YouTube
- **Canciones Más Gustadas**:
  - Ranking de las canciones catalanas más populares
  - Visualización del número de "likes" para cada canción
  - Destacado especial para las 3 primeras posiciones
  - Enlaces directos a YouTube para escuchar las canciones
- **Sobre Nosotros**:
  - Información sobre el proyecto universitario
  - Sección de contacto con enlace mailto

## Características técnicas

- Interfaz responsiva con Tailwind CSS
- Animaciones y transiciones fluidas
- Visualización de datos interactiva
- Efectos visuales relacionados con música
- Integración con la API de YouTube para reproducción de videos
- Sistema de extracción de IDs de videos de YouTube para seguimiento de valoraciones
- Expansión/contracción de videos al hacer clic en las recomendaciones
- Lectura y procesamiento de datos desde archivos CSV

## Sistema de valoración

La aplicación permite valorar las recomendaciones musicales mediante un sistema de estrellas:

1. Cada recomendación tiene un botón "Valorar" que abre un modal
2. El usuario puede asignar de 1 a 5 estrellas según la calidad de la recomendación
3. Una vez valorada, la recomendación queda marcada como "Valorado" hasta refrescar la página
4. El sistema registra el ID del video que buscó el usuario y el ID del video recomendado
5. Los datos de valoración pueden integrarse con un sistema de backend para mejorar las recomendaciones futuras

## Estructura del proyecto

```
interface/
├── app.py                  # Aplicación principal de Flask
├── static/                 # Archivos estáticos
│   ├── css/                # Estilos CSS
│   ├── js/                 # Scripts JavaScript
│   ├── images/             # Imágenes (incluye bg.png)
│   └── csv/                # Archivos CSV con datos de canciones
└── templates/              # Plantillas HTML
    ├── index.html          # Página inicial
    ├── loading.html        # Pantalla de carga
    ├── results.html        # Página de resultados
    ├── library.html        # Biblioteca de canciones
    ├── liked.html          # Canciones más gustadas
    └── about.html          # Página sobre nosotros
```

## Personalización

La aplicación utiliza una imagen de fondo personalizable (`bg.png`) ubicada en la carpeta `static/images/`. Puedes reemplazarla por cualquier imagen de tu elección. 

## Mejoras técnicas

- **Extracción de IDs de YouTube**: El sistema utiliza múltiples métodos para extraer correctamente el ID de video de diferentes formatos de URL de YouTube.
- **Reproducción de videos**: Los videos se muestran expandiéndose hacia abajo cuando se hace clic en una recomendación y se cierran automáticamente al abrir otro video.
- **Integración con sistemas externos**: La función de valoración está preparada para integrarse con sistemas externos mediante una función `registrarValoracion(valoracion, linkUsuario, linkRecomendacion)`.
- **Gestión de datos CSV**: La aplicación lee y procesa datos de canciones desde archivos CSV, permitiendo una fácil actualización del catálogo musical.

## Notas adicionales

- La interfaz de usuario está completamente en catalán.
- Copyright © 2025 Em Sona - Todos los derechos reservados. 
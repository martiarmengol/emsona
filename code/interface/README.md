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

## Características técnicas

- Interfaz responsiva con Tailwind CSS
- Animaciones y transiciones fluidas
- Visualización de datos interactiva
- Efectos visuales relacionados con música

## Estructura del proyecto

```
interface/
├── app.py                  # Aplicación principal de Flask
├── static/                 # Archivos estáticos
│   ├── css/                # Estilos CSS
│   ├── js/                 # Scripts JavaScript
│   └── images/             # Imágenes (incluye bg.png)
└── templates/              # Plantillas HTML
    ├── index.html          # Página inicial
    ├── loading.html        # Pantalla de carga
    └── results.html        # Página de resultados
```

## Personalización

La aplicación utiliza una imagen de fondo personalizable (`bg.png`) ubicada en la carpeta `static/images/`. Puedes reemplazarla por cualquier imagen de tu elección. 
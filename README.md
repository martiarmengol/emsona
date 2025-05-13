## Project Description & Planning
**Project:** *Catalan Music Classification and Analysis*   
**Contributors:** Adrià Cortés Cugat, Martí Armengol Ayala, Sergi De Vicente, Guillem Garcia, Jofre Geli, Javier Echávarri, Alex Salas Molina  
**Course:** Music Technology

# **- Why:** 
To explore and understand how Catalan music has evolved over time by analyzing differences in musical features between bands and eras. The project will highlight stylistic patterns and provide insight into genre, instrumentation, and production trends in the Catalan music scene.
**- What:** We aim to create a machine learning pipeline that classifies songs based on audio features and visualizes their relationships using dimensionality reduction. The system will analyze a curated dataset of 120 songs from 12 bands, split across two distinct time periods.
**- Who:** The project is developed by a team of six members: Adrià Cortés Cugat, Martí Armengol Ayala, Sergi de Vicente, Guillem Garcia, Jofre Geli and Javier Echavarri. Each team member is responsible for a specific aspect of the project, ranging from data collection and feature extraction to ML modeling and visualization.
**- When:** The project will be developed over a 10-week period during the trimester, with structured weekly milestones including dataset preparation, ML experimentation, state-of-the-art research, and final presentation.

# **1 Introduction**

Catalan music, with its rich cultural heritage and evolving stylistic expressions, offers a unique landscape for computational analysis. In this project, we aim to explore how musical trends and band-specific characteristics have changed over time by developing a classification and analysis pipeline based on machine learning techniques. The inspiration behind this project stems from our interest in understanding the evolution of music through data: can we quantify how sound has changed across decades? Can algorithms capture and reveal stylistic patterns that might not be immediately obvious to human listeners?To answer these questions, we will create and analyze a curated dataset of 120 songs by 12 different Catalan bands, carefully selected to represent two distinct time periods—songs released before 2012 and songs released after 2018. This temporal division allows us to investigate shifts in musical features and production choices across more than a decade of artistic development.Each song in the dataset will be annotated or processed to extract meaningful musical features such as instrumentation, genre, acoustic versus electronic nature, gender of the voice, and tempo (BPM). These features, both manually curated and automatically extracted, will form the basis of our classification task. Additionally, we will compute audio embeddings using pre-trained deep learning models like CLAP or other state-of-the-art audio encoders. These embeddings are designed to capture complex acoustic patterns and will serve as input for clustering and dimensionality reduction techniques such as K-Means, PCA, UMAP, and t-SNE.The outputs of these analyses will not only support classification between songs or groups but also enable insightful visualizations that map the landscape of Catalan music over time. Through this project, we aim to build a system that is both technically robust and musically meaningful—one that enables interpretation, fosters understanding, and ultimately celebrates the diversity and evolution of Catalan musical identity.

## Environment Variables

Before running any scripts that interact with the Gemini API, ensure your API key is set as an environment variable:

```bash
export GENAI_API_KEY="your_actual_api_key"
```

## Software Specification Requirements
**Project:** *Catalan Music Classification and Analysis*  
**Version 1**  
**Contributors:** Adrià Cortés Cugat, Martí Armengol Ayala, Sergi De Vicente, Guillem Garcia, Jofre Geli, Javier Echávarri, Alex Salas Molina  
**Course:** Music Technology  

# Purpose

This document outlines the **software requirements** for the system developed as part of the project **“Classification and Analysis of Catalan Songs,”** carried out within the **Music Technology Lab** course. The system aims to **explore and classify stylistic patterns in Catalan music** by leveraging **audio feature extraction**, **embedding models**, **machine learning techniques**, and **data visualization tools**.

The project originates from a context where we are expected to apply concepts from **signal processing**, **artificial intelligence (machine learning)**, and **music information retrieval**, using **real-world datasets** with an emphasis on **collaborative software development** through tools like **Python**, **Git**, and **web technologies**. Specifically, the system focuses on analyzing a **curated collection of 120 Catalan songs**, spanning **multiple bands** and **two distinct time periods** (songs released **before 2012** and those **after 2018)**. The primary objective is to **detect stylistic evolutions over time** and understand how different **musical features** contribute to the **uniqueness of each band or era**.

At a high level, the system enables users (**researchers, students, or educators**) to **upload and annotate audio data**, **extract relevant musical features**, **compute embeddings using pretrained models**, and **visualize relationships between songs** using **clustering** and **dimensionality reduction** techniques. The software supports both **exploratory analysis** and the **evaluation of classification models**, helping users better understand **musical similarity** based on acoustic attributes such as **genre**, **BPM**, **instrumentation**, and **vocal characteristics**.

The main use cases include **preprocessing a collection of Catalan songs**, **computing embeddings** using audio encoders like **CLAP**, **applying unsupervised learning algorithms** to uncover stylistic patterns, and **generating interpretable visualizations** that reveal differences between bands or time periods. These functionalities are intended to support **research and academic exploration** of Catalan music through a **data-driven and reproducible framework**.


# Overall description

The system developed in this project is designed to support the **classification and analysis of Catalan music** using **machine learning** and **audio processing techniques**. Its primary function is to **extract meaningful features from songs**, **compute audio embeddings**, **apply unsupervised clustering**, and **visualize the results** to reveal patterns of similarity between **bands and time periods**. The system is built to be **modular**, **interpretable**, and **reproducible**, with all components integrated into a **coherent workflow**.

The **overall system architecture** is composed of **four main stages**. First, the **data ingestion and preprocessing module** handles **metadata management** and **audio file organization**. This module **standardizes input formats**, processes raw “_.mp3_” files, and prepares “_.csv_” metadata files containing features such as **genre**, **BPM**, **instrumentation type**, and **vocal characteristics**. Second, the **feature extraction and embedding module** uses **pretrained models** such as **CLAP** to transform each song into a **fixed-length vector representation** that captures its audio characteristics. Third, the **analysis module** applies **unsupervised learning algorithms** like **K-Means clustering** and dimensionality reduction methods such as **PCA**, **t-SNE**, or **UMAP**. These techniques reveal **latent structures** in the data and help explore **stylistic similarities**. Finally, the **visualization and interpretation module** generates **2D plots** of the audio embedding space, colored by relevant features like **time period** or **band identity**, and provides interpretation tools for understanding the results.

The **user interface** is currently **command-line based**, designed for **technical users** such as students or researchers familiar with **Python**. **Jupyter notebooks** and scripts allow users to execute each phase of the pipeline, from feature extraction to visualization, with clearly documented inputs and outputs. Users interact with the system by **running scripts**, **configuring parameters** (e.g., number of clusters or choice of embedding model), and **inspecting the generated visualizations** and “_.csv_” outputs.

From a **software design perspective**, the system adheres to a **modular architecture**. Code is organized into **separate components** for **data loading**, **feature extraction**, **clustering**, and **visualization**, making it **easy to test, maintain, and extend**. External libraries such as **NumPy**, **pandas**, **scikit-learn**, and **matplotlib** are used for **data processing** and **visualization**, while the **CLAP model** is integrated through **open-source APIs** or **pretrained checkpoints**.

The system operates under certain **constraints**. Due to **computational limits**, embeddings and clustering are performed on a **relatively small dataset** (120 songs). **Audio files** are stored locally or referenced via external links, as **repository size constraints** prevent direct hosting of large media files on GitHub. Another constraint is the **limited availability of labeled data**, which restricts the use of supervised learning and requires a focus on **unsupervised and exploratory techniques**.

In summary, the system is designed to be a **flexible**, **research-oriented platform** for exploring **Catalan music** through the lens of **data science** and **machine learning**. It emphasizes **modularity**, **interpretability**, and **academic rigor**, while remaining **accessible to users with basic coding experience**.


# Specific requirements

This section outlines the functional and non-functional requirements necessary for the implementation and execution of the music classification and analysis system. It includes both software functionalities and the technical environment in which the system operates.

 **Functional Requirements**

The system is expected to fulfill the following core functionalities:

- **Audio Preprocessing and Metadata Handling**\
  The system must be able to ingest audio files (e.g., .mp3 or .wav) and associate them with structured metadata, including features such as band name, year of release, genre, BPM, instrumentation type, and vocal gender. This metadata should be saved and organized in .csv files for later processing.

- **Audio Embedding Generation**\
  The system must support the use of a pre trained audio embedding model (e.g., CLAP) to convert each song into a fixed-size vector representation. This process should be automated and compatible with local audio files.

- **Unsupervised Clustering and Dimensionality Reduction**\
  The system must implement clustering algorithms (e.g., K-Means) to group songs based on their embeddings. It must also allow dimensionality reduction techniques (PCA, t-SNE, or UMAP) to project high-dimensional data into 2D or 3D space for visualization purposes.

- **Visualization of Results**\
  The system must generate interpretable plots that display clusters of songs, color-coded by attributes such as band, year, or genre. These plots must be exportable (e.g., as .png files) and support visual comparison between groups.

- **Modularity and Script-Based Execution**

Each functional component (e.g., feature extraction, clustering, visualization) must be implemented as a reusable script or module. Users should be able to execute each step independently and adjust parameters through configuration files or command-line arguments.

**Hardware Requirements**

The system is designed to run on standard consumer-level hardware with the following recommended specifications:

- **Processor**: Multi-core CPU (Intel i5/Ryzen 5 or higher)

- **RAM**: Minimum 8 GB (16 GB recommended for embedding and clustering tasks)

- **Storage**: At least 5 GB of free disk space for audio files and output artifacts

- **GPU**: Optional, but beneficial for accelerating embedding computation with certain models

The system has been developed and tested primarily on personal laptops and university workstations. GPU usage is not mandatory but may be integrated later for larger datasets.

**Software Requirements**

The software stack for the project includes the following dependencies and development tools:

- **Programming Language**: Python 3.8 or later

- **Libraries**: NumPy, pandas, scikit-learn, matplotlib, seaborn, librosa, umap-learn, plotly

- **Audio Models**: Integration with CLAP or similar pretrained audio embedding frameworks via available APIs or checkpoints

- **Environment Management:** Recommended use of virtual environments for dependency isolation

- **Repository Management**: Git and GitHub for version control and team collaboration

- **Notebook Support**: Jupyter Notebooks for testing and demonstration of system functionalities

**Optional Tools**: Google Drive or external storage for hosting large audio files outside GitHub.

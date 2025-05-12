## Software de development of tools
**Project:** *Catalan Music Classification and Analysis*   
**Contributors:** Adrià Cortés Cugat, Martí Armengol Ayala, Sergi De Vicente, Guillem Garcia, Jofre Geli, Javier Echávarri, Alex Salas Molina  
**Course:** Music Technology  

---

## Code Repository  

Link to our repository:
https://github.com/martiarmengol/MTG-102 

The project is managed collaboratively through a public GitHub repository, ensuring transparent access for all team members and instructors. The repository is structured to support modular development, version control, and reproducibility of all experiments and results.

- **code/**: Contains all Python scripts and Jupyter notebooks related to audio processing, embedding extraction (using CLAP), clustering algorithms (K-Means), classification (KNN), and data visualization (PCA, UMAP, t-SNE). It is organized to allow independent testing and reuse of each component.
- **database_csv/**: Stores .csv files with metadata and extracted audio features (BPM, instrumentation, genre, voice gender, acoustic v.s. Electronic, and YT Link). These structured datasets are used as inputs for machine learning models and for clustering/visualization tasks.
- **db_downloaded_songs/**: Contains the audio files used in the project. This folder includes organized .mp3 files grouped by time period. If hosting limitations apply, the repository links to external storage (Google Drive) while maintaining file structure and naming consistency.
- **documentation/**: Includes all written reports and supporting documents, such as the State of the Art, Software Development Tools, and others. It serves as a central location for academic and technical documentation
- **.gitattributes and .gitignore**: These configuration files manage cross-platform consistency (line endings) and ensure sensitive or unnecessary files (cache, large binaries) are excluded from version control. This supports clean collaboration and reproducible environments.

---

## Software Tools


### Audio Feature Extraction and Processing

- **Librosa**: 
A Python library for music and audio analysis. Used for extracting features such as MFCCs, spectral centroid, tempo, and beat-related information. It also supports preprocessing (e.g., resampling, trimming) and audio visualization.


- **Essentia**:
An open-source library developed by the Music Technology Group (UPF), offering a rich set of algorithms for audio feature extraction, classification, and segmentation. Used particularly for rhythm, tonal, and timbral analysis in early stages.


- **Sonic Visualiser**:
A visual tool used for manual inspection of waveforms, spectrograms, and annotations. Valuable for exploring audio characteristics and validating the dataset through visual inspection.

### Machine Learning & Statistical Analysis

- **Scikit-learn**:
A Python library for implementing classical machine learning algorithms such as K-Nearest Neighbors (KNN), K-Means clustering, PCA, and model evaluation. It integrates well with NumPy and audio features extracted from Librosa or CLAP.


- **CLAP (Contrastive Language-Audio Pretraining)**:
A pre-trained deep learning model used to convert audio into high-dimensional embeddings. These embeddings are used as feature vectors for classification and clustering. CLAP is central to the project’s embedding pipeline.


- **CLIP (Contrastive Language-Image Pretraining)**:
Although not used directly, CLIP serves as an architectural and methodological reference for CLAP. It highlights best practices in multimodal learning and zero-shot classification.


- **RStudio**:
Used for statistical analysis and visualization, particularly for clustering validation. R’s visualization packages provide complementary tools for interpreting high-dimensional embeddings and similarity metrics.

### Dimensionality Reduction & Visualization

- **PCA (Principal Component Analysis)**:
A linear method used to reduce feature dimensionality while retaining the most significant variance in the data. Applied for both preprocessing and visualization.


- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
A nonlinear technique for visualizing high-dimensional embeddings in 2D or 3D space, emphasizing local relationships between songs.


- **UMAP (Uniform Manifold Approximation and Projection)**:
A powerful nonlinear dimensionality reduction technique that preserves both local and global structure. Used for visualizing complex audio embedding spaces.

### Version Control & Collaboration

- **Git & GitHub**:
Git is used for version control, while GitHub is the collaborative platform that hosts the repository. The project repository contains all code, documentation, and configuration files, structured into modules for preprocessing, analysis, and reporting.

---

## Collaborative coding strategy

To ensure efficient teamwork, maintain code quality, and facilitate reproducibility, our group follows a structured collaborative coding strategy using Git and GitHub. These practices help coordinate efforts across team members, manage project complexity, and maintain a stable development workflow throughout the trimester.

### Branching Strategy

The development workflow is structured around a branching strategy in which the main branch holds the most stable and validated version of the codebase. Each new functionality or experiment is developed in a dedicated feature branch. For example, branches like feature/embedding, feature/classifier, or feature/visualization allow individual members to work independently on specific parts of the system without interfering with ongoing work by others. 

### Team Responsibilities

Each member of the group is responsible for a specific component of the project, such as embedding generation, clustering, visualization, or dataset management. While responsibilities are clearly distributed, collaboration and code sharing are encouraged during integration phases. This structured yet flexible collaboration model allows the team to work efficiently, maintain high standards, and deliver a cohesive and robust final system.

# DSAN6700 - Fashion Recommendation System

Team: Courtney Green, Li-Wen Hu, Satomi Ito, Nandini Kodali, Sophia Rutman

A fashion recommendation system that helps users style their own clothing or build new outfits from an uploaded image. The system combines computer vision (ResNet-50 embeddings) with efficient similarity search and recommendation algorithms.

## Project Structure

### Computer Vision Files (`ComputerVisionFiles/`)

- `ResNet.ipynb` - Colab notebook used to preprocess Fashion MNIST, extract embeddings using ResNet-50, and generate visualization outputs
- `fashion_mnist_classes.txt` - Text file listing the 10 Fashion MNIST class names
- `fashion_mnist_labels.npy` - Array of label indices for each embedded image (stored with Git LFS)
- `fashion_mnist_resnet50_embeddings.npy` - Array of 2048-dimensional ResNet-50 embeddings for the entire Fashion MNIST training set (stored with Git LFS)

### Recommendation Engine (`RecommendationFiles/`)

- `recommendation_engine.py` - Main recommendation engine module with PCA dimensionality reduction and FAISS similarity search
- `example_usage.py` - Example usage script demonstrating all features
- `benchmark.py` - Comprehensive benchmarking script
- `test_integration.py` - Integration tests
- `quick_test.py` - Quick verification script
- `run_tests.sh` - Automated test runner
- `requirements.txt` - Python dependencies
- `README.md` - Detailed documentation

## Quick Start

### 1. Install Dependencies

```bash
cd RecommendationFiles
pip install -r requirements.txt
```

```bash
cd backend
pip install -r requirements.txt
```

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Ensure Data Files Are Available

```bash
# From project root
git lfs install
git lfs pull
```

### 3. Run Quick Test

```bash
cd RecommendationFiles
python quick_test.py
```

## Getting Started

For detailed instructions, see the [Recommendation Engine README](RecommendationFiles/README.md).

## Git LFS

The `.npy` files are large and are therefore tracked using Git LFS instead of regular Git. 

After cloning the repository, run:

```bash
git lfs install
git lfs pull
```

## Project Components

### Computer Vision Component

The CV team provides ResNet-50 embeddings (2048-dimensional) extracted from Fashion MNIST images. These embeddings are stored in the `ComputerVisionFiles/` directory.

### Recommendation Engine Component

The recommendation engine provides:
- **Dimensionality Reduction**: PCA-based reduction from 2048D to configurable dimensions (default: 128D)
- **FAISS Integration**: High-performance similarity search
- **Flexible Filtering**: Filter by class, exclude items, combine filters
- **Performance Optimized**: ~10,000+ queries/second, ~30MB memory footprint

See [RecommendationFiles/README.md](RecommendationFiles/README.md) for complete documentation.

### Backend

The backend uses FastAPI to connect S3 and RDS services from AWS to our site. In order to run and test the API calls locally, run 

```bash
uvicorn backend.app.main:app --reload
```

Then, navigate to http://127.0.0.1:8000 on your machine to access specific calls. 

The backend is deployed on the cloud using ***Render***, located at https://dsan6700.onrender.com. 

### Frontend

The frontend is written using Streamlit, where the API calls are connected to the various functionalities. To run the frontend locally, run 
```bash
uvicorn backend.app.main:app --reload
```

The frontend is deployed on the cloud using Streamlit, located at https://stylesynth.streamlit.app. 




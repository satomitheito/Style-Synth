# DSAN6700


# Computer Vision Files

- `ResNet.ipynb` Colab notebook used to preprocess Fashion MNIST, extract embeddings using ResNet-50, and generate visualization outputs.

- `fashion_mnist_classes.txt` Text file listing the 10 Fashion MNIST class names.

- `fashion_mnist_labels.npy` (stored with Git LFS): Array of label indices for each embedded image.

- `fashion_mnist_resnet50_embeddings.npy` (stored with Git LFS): Array of 2048-dimensional ResNet-50 embeddings for the entire Fashion MNIST training set.

# Git LFS

The .npy files are large and are therefore tracked using Git LFS instead of regular Git. 

After cloning the repository, run:

```bash
git lfs install
git lfs pull
```
# Image Look-Alike Finder
**Linear Algebra Implementation for Image Similarity Search**

Course: UE24MA241B – Linear Algebra and Its Applications

## Project Overview

This project implements an image similarity search system using pure linear algebra concepts. Given a query image (face from the Olivetti dataset), the system finds the most visually similar images from a database - all without any machine learning, just linear algebra!

## Linear Algebra Pipeline

Each step maps directly to concepts from your linear algebra course:

### Step 1 — Real-World Data → Matrix Representation
- **Topics**: Matrices, Linear Transformations
- **Implementation**: Each 64×64 grayscale image is flattened into a 4096-dimensional vector
- **Result**: N × 4096 data matrix where each row represents an image

### Step 2 — Matrix Simplification  
- **Topics**: Gaussian Elimination, RREF
- **Implementation**: Mean-center the matrix and compute RREF to analyze rank
- **Result**: Centered matrix with duplicate/near-identical images identified

### Step 3 — Structure of the Space
- **Topics**: Vector Spaces, Subspaces, Basis, Rank & Nullity
- **Implementation**: Compute covariance matrix AᵀA and analyze its properties
- **Result**: Understanding of representable pixel patterns and redundancies

### Step 4 — Remove Redundancy
- **Topics**: Linear Independence, Basis Selection
- **Implementation**: Extract linearly independent set of image vectors
- **Result**: Clean basis for the image space without duplicates

### Step 5 — Orthogonalization
- **Topics**: Gram-Schmidt, Orthogonal Bases
- **Implementation**: Apply Gram-Schmidt to create orthogonal "eigenface-like" directions
- **Result**: Orthogonal basis vectors capturing distinct visual patterns

### Step 6 — Projection
- **Topics**: Orthogonal Projections, Projection onto Subspaces
- **Implementation**: Project images onto orthogonal basis for low-dimensional coordinates
- **Result**: "Visual fingerprints" for each image

### Step 7 — Prediction/Approximation (Similarity)
- **Topics**: Least Squares Solution, x̂ = (AᵀA)⁻¹Aᵀb
- **Implementation**: Use least squares to find best linear combination approximation
- **Result**: Similarity scores based on reconstruction error

### Step 8 — Pattern Discovery
- **Topics**: Eigenvalues & Eigenvectors
- **Implementation**: Compute eigenvalues/eigenvectors of covariance matrix
- **Result**: Principal directions showing dominant visual patterns

### Step 9 — System Simplification
- **Topics**: Diagonalization, Symmetric Matrix Diagonalization
- **Implementation**: Diagonalize and keep top-k eigenvalues/vectors
- **Result**: Fast, noise-free similarity search in reduced space

## Installation

1. Clone or download the project files
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Command Line Demo
```bash
python image_lookalike_finder.py
```
This runs the complete pipeline and shows results for a sample query image.

### Method 2: Interactive Web Interface
```bash
python demo.py
```
This launches a Gradio web interface where you can:
- Select any query image from the dataset
- View top 3 similar matches with similarity scores
- See the mean face and eigenfaces (discovered patterns)

### Method 3: Run Tests
```bash
python test_pipeline.py
```
This tests each step of the pipeline to ensure correctness.

## Project Structure

```
├── image_lookalike_finder.py    # Main implementation with all 9 steps
├── demo.py                     # Interactive Gradio web interface
├── test_pipeline.py            # Test suite for validation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Key Features

- **Pure Linear Algebra**: No scikit-learn ML, no PyTorch, no CNNs
- **Educational**: Each step maps to course concepts for viva preparation
- **Visual**: See eigenfaces, mean face, and similarity results
- **Interactive**: Web interface for easy exploration
- **Tested**: Comprehensive test suite validates each mathematical step

## Mathematical Concepts Demonstrated

1. **Matrix Operations**: Flattening, mean-centering, covariance computation
2. **Linear Independence**: QR decomposition for basis selection
3. **Orthogonality**: Gram-Schmidt process (via QR for stability)
4. **Projections**: Orthogonal projection onto subspaces
5. **Least Squares**: x̂ = (AᵀA)⁻¹Aᵀb formulation
6. **Eigen Decomposition**: Pattern discovery and diagonalization
7. **Dimensionality Reduction**: Principal component analysis

## Dataset

Uses the Olivetti faces dataset from scikit-learn:
- 400 face images
- 64×64 pixels each
- 4096-dimensional vectors when flattened
- Various lighting conditions and facial expressions

## Example Output

When you run the demo, you'll see:
- **Query Image**: The selected face
- **Mean Face**: Average of all faces in dataset
- **Top 3 Matches**: Most similar faces with similarity scores
- **Eigenfaces**: Top 4 principal components (discovered patterns)

## Academic Integrity

This implementation follows the project guidelines strictly:
- ✅ Uses only NumPy for mathematical operations
- ✅ No machine learning libraries for core algorithm
- ✅ Each step explicitly implements linear algebra concepts
- ✅ SymPy used only for RREF computation
- ✅ scikit-learn used only for dataset access

## Perfect for Viva Preparation

The code is structured to help you explain each step:
- Clear function names matching the pipeline steps
- Detailed comments explaining the mathematics
- Print statements showing intermediate results
- Visualizations for intuitive understanding

## Performance

- **Fast**: Matrix operations optimized with NumPy
- **Memory Efficient**: Processes 400×4096 matrices efficiently
- **Scalable**: Can handle larger image datasets
- **Accurate**: Mathematically sound implementation

---

**Course**: UE24MA241B – Linear Algebra and Its Applications  
**Project**: Image Look-Alike Finder using Pure Linear Algebra

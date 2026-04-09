"""
Image Look-Alike Finder - Linear Algebra Implementation
Course: UE24MA241B – Linear Algebra and Its Applications

This project implements image similarity search using pure linear algebra concepts.
Each step maps directly to linear algebra topics covered in the course.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class ImageLookalikeFinder:
    """
    Main class implementing the image similarity pipeline using linear algebra.
    """
    
    def __init__(self, n_components: int = 50):
        """
        Initialize the Image Lookalike Finder.
        
        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.faces_data = None
        self.faces_matrix = None
        self.mean_face = None
        self.centered_matrix = None
        self.covariance_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.orthogonal_basis = None
        self.projection_matrix = None
        
    def step1_load_and_create_matrix(self) -> np.ndarray:
        """
        Step 1: Real-World Data → Matrix Representation
        Load Olivetti faces dataset and create data matrix.
        
        Topics: Matrices, Linear Transformations
        """
        print("Step 1: Loading dataset and creating matrix representation...")
        
        # Load Olivetti faces dataset
        faces = fetch_olivetti_faces()
        self.faces_data = faces.images  # Shape: (400, 64, 64)
        
        # Flatten each image to create data matrix
        n_images = self.faces_data.shape[0]
        image_size = self.faces_data.shape[1] * self.faces_data.shape[2]
        
        # Create N × 4096 data matrix where each row is a flattened image
        self.faces_matrix = self.faces_data.reshape(n_images, image_size)
        
        print(f"Loaded {n_images} images of size {self.faces_data.shape[1]}×{self.faces_data.shape[2]}")
        print(f"Data matrix shape: {self.faces_matrix.shape}")
        
        return self.faces_matrix
    
    def step2_mean_center_and_rref(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 2: Matrix Simplification
        Mean-center the matrix and compute RREF to examine rank.
        
        Topics: Gaussian Elimination, RREF
        """
        print("\nStep 2: Mean-centering and RREF analysis...")
        
        # Compute mean face (average image)
        self.mean_face = np.mean(self.faces_matrix, axis=0)
        
        # Mean-center the matrix
        self.centered_matrix = self.faces_matrix - self.mean_face
        
        print(f"Mean face computed and subtracted from all images")
        print(f"Centered matrix shape: {self.centered_matrix.shape}")
        
        # Compute RREF for rank analysis (using a smaller sample for efficiency)
        print("Computing RREF to analyze rank...")
        sample_size = min(20, self.centered_matrix.shape[0])
        sample_matrix = self.centered_matrix[:sample_size, :]
        
        # Use NumPy for fast numerical rank analysis
        rank = np.linalg.matrix_rank(sample_matrix)
        nullity = sample_matrix.shape[1] - rank

        print(f"Sample matrix rank: {rank}")
        print(f"Sample matrix nullity: {nullity}")
        print(f"Number of pivot columns: {rank}")
        
        return self.centered_matrix, self.mean_face
    
    def step3_covariance_analysis(self) -> np.ndarray:
        """
        Step 3: Structure of the Space
        Compute and analyze the covariance matrix.
        
        Topics: Vector Spaces, Subspaces, Basis, Rank & Nullity
        """
        print("\nStep 3: Covariance matrix analysis...")
        
        # Compute covariance matrix AᵀA
        self.covariance_matrix = np.dot(self.centered_matrix.T, self.centered_matrix)
        
        # Analyze properties
        rank = np.linalg.matrix_rank(self.covariance_matrix)
        # Use efficient eigenvalue computation optimized for symmetric matrices
        eigenvals = np.linalg.eigvalsh(self.covariance_matrix)
        nullity = len(eigenvals) - np.sum(eigenvals > 1e-10)
        
        print(f"Covariance matrix shape: {self.covariance_matrix.shape}")
        print(f"Covariance matrix rank: {rank}")
        print(f"Covariance matrix nullity: {nullity}")
        print(f"Trace (total variance): {np.trace(self.covariance_matrix):.2f}")
        
        return self.covariance_matrix
    
    def step4_remove_linearly_dependent(self, tolerance: float = 1e-10) -> Tuple[np.ndarray, List[int]]:
        """
        Step 4: Remove Redundancy
        Extract linearly independent set of image vectors.
        
        Topics: Linear Independence, Basis Selection
        """
        print("\nStep 4: Removing linearly dependent images...")
        
        # Use NumPy QR decomposition — fast numerical computation
        # Transpose: each image vector becomes a column
        Q, R = np.linalg.qr(self.centered_matrix.T)

        # Diagonal entries of R reveal linearly independent columns
        independent_indices = sorted(np.where(np.abs(np.diag(R)) > tolerance)[0].tolist())
        
        # Extract independent images
        independent_matrix = self.centered_matrix[independent_indices, :]
        
        print(f"Original images: {self.centered_matrix.shape[0]}")
        print(f"Linearly independent images: {len(independent_indices)}")
        print(f"Removed {self.centered_matrix.shape[0] - len(independent_indices)} dependent images")
        
        return independent_matrix, independent_indices
    
    def step5_gram_schmidt_orthogonalization(self, matrix: np.ndarray) -> np.ndarray:
        """
        Step 5: Gram-Schmidt Orthogonalization
        Apply Gram-Schmidt process (via QR decomposition) to create orthogonal basis.
        
        Topics: Gram-Schmidt, Orthogonal Bases
        """
        print("\nStep 5: Gram-Schmidt orthogonalization...")
        
        # Apply Gram-Schmidt orthogonalization using QR decomposition
        # QR decomposition is numerically equivalent to Gram-Schmidt but more stable
        # Transpose matrix so columns are vectors to orthogonalize
        Q, R = np.linalg.qr(matrix.T, mode='reduced')
        
        # Q contains orthonormal basis vectors (columns)
        # Keep only the first n_components basis vectors
        n_basis_vectors = min(self.n_components, Q.shape[1])
        self.orthogonal_basis = Q[:, :n_basis_vectors]
        
        print(f"Orthogonal basis shape: {self.orthogonal_basis.shape}")
        print(f"Number of orthogonal directions: {self.orthogonal_basis.shape[1]}")
        
        # Verify orthogonality: QᵀQ should equal Identity matrix
        dot_product = np.dot(self.orthogonal_basis.T, self.orthogonal_basis)
        is_orthogonal = np.allclose(dot_product, np.eye(dot_product.shape[0]), atol=1e-10)
        print(f"Basis is orthogonal: {is_orthogonal}")
        
        return self.orthogonal_basis
    
    def step6_projection_onto_basis(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Step 6: Projection
        Project images onto the orthogonal basis.
        
        Topics: Orthogonal Projections, Projection onto Subspaces
        """
        print("\nStep 6: Projecting images onto orthogonal basis...")
        
        # Compute projection matrix P = Q(QᵀQ)⁻¹Qᵀ
        # Since Q is orthogonal, QᵀQ = I, so P = QQᵀ
        self.projection_matrix = np.dot(self.orthogonal_basis, self.orthogonal_basis.T)
        
        # Project all images onto the orthogonal basis
        # Coordinates = Qᵀx (since Q is orthogonal)
        projected_coordinates = np.dot(data_matrix, self.orthogonal_basis)
        
        print(f"Projected coordinates shape: {projected_coordinates.shape}")
        print(f"Dimensionality reduced from {data_matrix.shape[1]} to {projected_coordinates.shape[1]}")
        
        return projected_coordinates
    
    def step7_cosine_similarity(self, query_image: np.ndarray, database_projections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 7: Similarity Search using Cosine Similarity
        Use cosine similarity to find best matches in projected space.
        
        Topics: Cosine Similarity, Vector Projections
        """
        print("\nStep 7: Computing similarity using cosine similarity...")
        
        # Project query image onto the same basis
        query_projection = np.dot(query_image - self.mean_face, self.orthogonal_basis)
        
        # Compute cosine similarity between query and all database images
        # Cosine similarity = (a · b) / (||a|| * ||b||)
        similarities = []
        for db_proj in database_projections:
            # Compute dot product
            dot_product = np.dot(query_projection, db_proj)
            
            # Compute magnitudes
            query_norm = np.linalg.norm(query_projection)
            db_norm = np.linalg.norm(db_proj)
            
            # Avoid division by zero
            if query_norm > 0 and db_norm > 0:
                similarity = dot_product / (query_norm * db_norm)
            else:
                similarity = 0.0
            
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Find top images, filtering out perfect matches (similarity >= 0.999)
        # This handles both exact matches and near-perfect matches due to floating point precision
        valid_indices = np.where(similarities < 0.999)[0]
        valid_similarities = similarities[valid_indices]
        
        # Get top 3 most similar images from valid matches
        if len(valid_indices) >= 3:
            top_valid_indices = np.argsort(valid_similarities)[::-1][:3]
            top_indices = valid_indices[top_valid_indices]
            top_similarities = valid_similarities[top_valid_indices]
        else:
            # If fewer than 3 valid matches, take what we have
            top_valid_indices = np.argsort(valid_similarities)[::-1]
            top_indices = valid_indices[top_valid_indices]
            top_similarities = valid_similarities[top_valid_indices]
        
        print(f"Top 3 matches found with cosine similarities: {top_similarities}")
        print(f"Similarity scores range: [{np.min(similarities):.3f}, {np.max(similarities):.3f}]")
        
        return top_indices, top_similarities
    
    def step8_eigen_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 8: Pattern Discovery
        Compute eigenvalues and eigenvectors of covariance matrix.
        
        Topics: Eigenvalues & Eigenvectors
        """
        print("\nStep 8: Eigenvalue and eigenvector analysis...")
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        print(f"Computed {len(self.eigenvalues)} eigenvalues and eigenvectors")
        print(f"Top 5 eigenvalues: {self.eigenvalues[:5]}")
        print(f"Total variance explained by top {self.n_components} components: "
              f"{np.sum(self.eigenvalues[:self.n_components]) / np.sum(self.eigenvalues) * 100:.2f}%")
        
        return self.eigenvalues, self.eigenvectors
    
    def step9_diagonalization_and_reduction(self, k: int = None) -> np.ndarray:
        """
        Step 9: System Simplification
        Diagonalize covariance matrix and reduce dimensionality.
        
        Topics: Diagonalization, Symmetric Matrix Diagonalization
        """
        print("\nStep 9: Diagonalization and dimensionality reduction...")
        
        if k is None:
            k = self.n_components
        
        # Keep only top-k eigenvectors
        top_eigenvectors = self.eigenvectors[:, :k]
        top_eigenvalues = self.eigenvalues[:k]
        
        # Diagonalize: D = QᵀAQ where Q contains eigenvectors
        # For symmetric matrix, this gives us the principal components
        diagonalized = np.dot(top_eigenvectors.T, np.dot(self.covariance_matrix, top_eigenvectors))
        
        # Verify diagonalization
        is_diagonal = np.allclose(diagonalized, np.diag(np.diag(diagonalized)), atol=1e-10)
        print(f"Matrix successfully diagonalized: {is_diagonal}")
        
        # Project all images into reduced k-dimensional space
        reduced_projections = np.dot(self.centered_matrix, top_eigenvectors)
        
        print(f"Reduced to {k}-dimensional space")
        print(f"Reduced projections shape: {reduced_projections.shape}")
        
        return reduced_projections
    
    def visualize_results(self, query_idx: int = 0, top_indices: np.ndarray = None, similarity_scores: np.ndarray = None):
        """
        Visualize the query image, top matches, and eigenfaces.
        """
        if top_indices is None or similarity_scores is None:
            return
            
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # Handle case where we have fewer than 3 matches
        num_matches = min(3, len(top_indices), len(similarity_scores))
        
        # Query image
        axes[0, 0].imshow(self.faces_data[query_idx], cmap='gray')
        axes[0, 0].set_title('Query Image')
        axes[0, 0].axis('off')
        
        # Mean face
        axes[0, 1].imshow(self.mean_face.reshape(64, 64), cmap='gray')
        axes[0, 1].set_title('Mean Face')
        axes[0, 1].axis('off')
        
        # Top 3 matches
        for i in range(num_matches):
            if i+2 <= 4:  # Make sure we don't go beyond column 4
                idx = top_indices[i]
                score = similarity_scores[i]
                axes[0, i+2].imshow(self.faces_data[idx], cmap='gray')
                axes[0, i+2].set_title(f'Match {i+1}: {score:.3f}')
                axes[0, i+2].axis('off')
        
        # Top 4 eigenfaces - centered in 2nd row with equal margins
        eigenface_positions = [0, 1, 3, 4]  # Positions 0,1,3,4 to center 4 images in 5-column grid
        for i, pos in enumerate(eigenface_positions):
            eigenface = self.eigenvectors[:, i].reshape(64, 64)
            axes[1, pos].imshow(eigenface, cmap='gray')
            axes[1, pos].set_title(f'Eigenface {i+1}\nlambda = {self.eigenvalues[i]:.2f}')
            axes[1, pos].axis('off')
        
        # Leave column 2 (middle column) blank for centering
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

    def run_complete_pipeline(self, query_idx: int = 0):
        """
        Run the complete image similarity pipeline.
        """
        print("=" * 60)
        print("IMAGE LOOK-ALIKE FINDER - LINEAR ALGEBRA PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and create matrix
        self.step1_load_and_create_matrix()
        
        # Step 2: Mean-center and RREF
        centered_matrix, mean_face = self.step2_mean_center_and_rref()
        
        # Step 3: Covariance analysis
        covariance_matrix = self.step3_covariance_analysis()
        
        # Step 4: Remove linearly dependent
        independent_matrix, independent_indices = self.step4_remove_linearly_dependent()
        
        # Step 5: Gram-Schmidt orthogonalization
        orthogonal_basis = self.step5_gram_schmidt_orthogonalization(independent_matrix)
        
        # Step 6: Projection
        database_projections = self.step6_projection_onto_basis(self.centered_matrix)
        
        # Step 7: Cosine similarity
        query_image = self.faces_matrix[query_idx]
        top_indices, similarity_scores = self.step7_cosine_similarity(query_image, database_projections)
        
        # top_indices are indices into the full database
        original_top_indices = top_indices.tolist()
        
        # Step 8: Eigen analysis
        eigenvalues, eigenvectors = self.step8_eigen_analysis()
        
        # Step 9: Diagonalization
        reduced_projections = self.step9_diagonalization_and_reduction()
        
        # Visualize results
        self.visualize_results(query_idx, np.array(original_top_indices), similarity_scores)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return original_top_indices, similarity_scores

def main():
    """
    Main function to demonstrate the image look-alike finder.
    """
    # Create instance
    finder = ImageLookalikeFinder(n_components=50)
    
    # Run complete pipeline with a sample query
    query_index = 10  # You can change this to test different queries
    top_matches, scores = finder.run_complete_pipeline(query_idx=query_index)
    
    print(f"\nQuery image index: {query_index}")
    print("Top 3 most similar images:")
    for i, (idx, score) in enumerate(zip(top_matches, scores)):
        print(f"  {i+1}. Image {idx} - Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
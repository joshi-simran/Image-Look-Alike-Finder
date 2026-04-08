"""
Demo script for Image Look-Alike Finder with Gradio UI
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from image_lookalike_finder import ImageLookalikeFinder
from sklearn.datasets import fetch_olivetti_faces

# Initialize the finder
finder = ImageLookalikeFinder(n_components=50)

def initialize_system():
    """Initialize the complete pipeline"""
    try:
        # Run the complete pipeline once
        faces = fetch_olivetti_faces()
        finder.faces_data = faces.images
        finder.faces_matrix = faces.data
        finder.mean_face = np.mean(finder.faces_matrix, axis=0)
        finder.centered_matrix = finder.faces_matrix - finder.mean_face
        finder.covariance_matrix = np.dot(finder.centered_matrix.T, finder.centered_matrix)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(finder.covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        finder.eigenvalues = eigenvalues[idx]
        finder.eigenvectors = eigenvectors[:, idx]
        
        # Use top eigenvectors as orthogonal basis
        finder.orthogonal_basis = finder.eigenvectors[:, :50]
        
        # Project all images
        finder.database_projections = np.dot(finder.centered_matrix, finder.orthogonal_basis)
        
        return "System initialized successfully!"
    except Exception as e:
        return f"Error initializing system: {str(e)}"

def find_similar_images(query_idx):
    """Find similar images for the given query index"""
    try:
        if query_idx >= len(finder.faces_data):
            return None, "Invalid query index!"
        
        query_image = finder.faces_matrix[query_idx]
        
        # Compute similarity scores
        query_projection = np.dot(query_image - finder.mean_face, finder.orthogonal_basis)
        errors = np.sum((query_projection - finder.database_projections) ** 2, axis=1)
        top_indices = np.argsort(errors)[:3]
        similarity_scores = 1.0 / (1.0 + errors[top_indices])
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Query image
        axes[0, 0].imshow(finder.faces_data[query_idx], cmap='gray')
        axes[0, 0].set_title('Query Image')
        axes[0, 0].axis('off')
        
        # Mean face
        axes[0, 1].imshow(finder.mean_face.reshape(64, 64), cmap='gray')
        axes[0, 1].set_title('Mean Face')
        axes[0, 1].axis('off')
        
        # Top 3 matches
        for i in range(min(3, len(top_indices))):
            if i+2 < 4:  # Make sure we don't go beyond column 3
                idx = top_indices[i]
                score = similarity_scores[i]
                axes[0, i+2].imshow(finder.faces_data[idx], cmap='gray')
                axes[0, i+2].set_title(f'Match {i+1}: {score:.3f}')
                axes[0, i+2].axis('off')
        
        # Top 4 eigenfaces
        for i in range(4):
            eigenface = finder.eigenvectors[:, i].reshape(64, 64)
            axes[1, i].imshow(eigenface, cmap='gray')
            axes[1, i].set_title(f'Eigenface {i+1}\nλ = {finder.eigenvalues[i]:.2f}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save plot to temporary file
        import io
        from PIL import Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        # Create results text
        results_text = f"Query Image: {query_idx}\n"
        results_text += "Top 3 Most Similar Images:\n"
        for i, (idx, score) in enumerate(zip(top_indices, similarity_scores)):
            results_text += f"  {i+1}. Image {idx} - Similarity: {score:.4f}\n"
        
        return result_image, results_text
        
    except Exception as e:
        return None, f"Error finding similar images: {str(e)}"

# Initialize the system
status_message = initialize_system()

# Create Gradio interface
with gr.Blocks(title="Image Look-Alike Finder - Linear Algebra") as demo:
    gr.Markdown("# Image Look-Alike Finder")
    gr.Markdown("## Linear Algebra Implementation for Image Similarity Search")
    gr.Markdown(f"**Status:** {status_message}")
    
    with gr.Row():
        with gr.Column():
            query_slider = gr.Slider(
                minimum=0, 
                maximum=399, 
                value=10, 
                step=1,
                label="Select Query Image Index (0-399)"
            )
            find_button = gr.Button("Find Similar Images", variant="primary")
        
        with gr.Column():
            gr.Markdown("### Instructions:")
            gr.Markdown("1. Select a query image using the slider")
            gr.Markdown("2. Click 'Find Similar Images' to search")
            gr.Markdown("3. View results showing top 3 matches")
            gr.Markdown("4. Eigenfaces show discovered patterns")
    
    with gr.Row():
        result_image = gr.Image(label="Results Visualization", type="pil")
        results_text = gr.Textbox(label="Similarity Results", lines=6)
    
    # Event handlers
    find_button.click(
        fn=find_similar_images,
        inputs=[query_slider],
        outputs=[result_image, results_text]
    )
    
    # Examples
    gr.Examples(
        examples=[
            [10], [50], [100], [150], [200], [250], [300], [350]
        ],
        inputs=[query_slider]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)

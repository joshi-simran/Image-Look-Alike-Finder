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
        
        # Create data matrix the same way as image_lookalike_finder.py
        n_images = finder.faces_data.shape[0]
        image_size = finder.faces_data.shape[1] * finder.faces_data.shape[2]
        finder.faces_matrix = finder.faces_data.reshape(n_images, image_size)
        finder.mean_face = np.mean(finder.faces_matrix, axis=0)
        finder.centered_matrix = finder.faces_matrix - finder.mean_face
        finder.covariance_matrix = np.dot(finder.centered_matrix.T, finder.centered_matrix)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(finder.covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        finder.eigenvalues = eigenvalues[idx]
        finder.eigenvectors = eigenvectors[:, idx]
        
        # Use Gram-Schmidt orthogonalization like the main file
        # Apply QR decomposition to get orthogonal basis
        Q, R = np.linalg.qr(finder.centered_matrix.T, mode='reduced')
        finder.orthogonal_basis = Q[:, :50]
        
        # Project all images using eigenvectors for better visual similarity
        finder.database_projections = np.dot(finder.centered_matrix, finder.eigenvectors[:, :50])
        
        return "System initialized successfully!"
    except Exception as e:
        return f"Error initializing system: {str(e)}"

def show_current_image(query_idx):
    """Show the currently selected image from slider"""
    try:
        if query_idx >= len(finder.faces_data):
            return None, f"Invalid index! Please select between 0 and {len(finder.faces_data)-1}"
        
        # Create a simple plot showing just the current image
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(finder.faces_data[query_idx], cmap='gray')
        ax.set_title(f'Selected Image #{query_idx}', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Save plot to temporary file
        import io
        from PIL import Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        info_text = f"Selected Image Index: {query_idx}\n"
        info_text += f"Image Shape: {finder.faces_data[query_idx].shape}\n"
        info_text += f"Database Size: {len(finder.faces_data)} images"
        
        return result_image, info_text
        
    except Exception as e:
        return None, f"Error displaying image: {str(e)}"

def find_similar_images(query_idx):
    """Find similar images for the given query index"""
    try:
        if query_idx >= len(finder.faces_data):
            return None, "Invalid query index!"
        
        query_image = finder.faces_matrix[query_idx]
        
        # Use eigenvectors for better visual similarity (while keeping Gram-Schmidt for pipeline)
        query_projection = np.dot(query_image - finder.mean_face, finder.eigenvectors[:, :50])
        
        # Calculate cosine similarities
        similarities = []
        for db_proj in finder.database_projections:
            dot_product = np.dot(query_projection, db_proj)
            query_norm = np.linalg.norm(query_projection)
            db_norm = np.linalg.norm(db_proj)
            
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
            similarity_scores = valid_similarities[top_valid_indices]
        else:
            # If fewer than 3 valid matches, take what we have
            top_valid_indices = np.argsort(valid_similarities)[::-1]
            top_indices = valid_indices[top_valid_indices]
            similarity_scores = valid_similarities[top_valid_indices]
        
        # Create visualization
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
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
            if i+2 <= 4:  # Make sure we don't go beyond column 4
                idx = top_indices[i]
                score = similarity_scores[i]
                axes[0, i+2].imshow(finder.faces_data[idx], cmap='gray')
                axes[0, i+2].set_title(f'Match {i+1}: {score:.3f}')
                axes[0, i+2].axis('off')
        
        # Top 4 eigenfaces - centered in 2nd row with equal margins
        eigenface_positions = [0, 1, 3, 4]  # Positions 0,1,3,4 to center 4 images in 5-column grid
        for i, pos in enumerate(eigenface_positions):
            eigenface = finder.eigenvectors[:, i].reshape(64, 64)
            axes[1, pos].imshow(eigenface, cmap='gray')
            axes[1, pos].set_title(f'Eigenface {i+1}\nlambda = {finder.eigenvalues[i]:.2f}')
            axes[1, pos].axis('off')
        
        # Leave column 2 (middle column) blank for centering
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot to temporary file
        import io
        from PIL import Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        # Create clean results text
        results_text = f"Query Image: #{query_idx}\n\n"
        results_text += "Top 3 Most Similar Images:\n"
        for i, (idx, score) in enumerate(zip(top_indices, similarity_scores)):
            results_text += f"  {i+1}. Image {idx:3d} - Similarity: {score:.4f} ({score*100:.1f}%)\n"
        
        results_text += f"\nDatabase: {len(finder.faces_data)} images\n"
        results_text += f"Accuracy Range: {np.min(similarities):.3f} - {np.max(similarities):.3f}"
        
        return result_image, results_text
        
    except Exception as e:
        return None, f"Error finding similar images: {str(e)}"

# Initialize the system
status_message = initialize_system()

# Create improved Gradio interface
with gr.Blocks(title="Image Look-Alike Finder - Linear Algebra", theme=gr.themes.Soft(), css="""
.center-text {
    text-align: center !important;
}
""") as demo:
    gr.Markdown("""
    # Image Look-Alike Finder
    """, elem_classes=["center-text"])
    
    with gr.Tabs():
        with gr.TabItem("Image Selection"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Select Query Image")
                    query_slider = gr.Slider(
                        minimum=0, 
                        maximum=399, 
                        value=10, 
                        step=1,
                        label="Image Index (0-399)",
                        info="Move slider to preview different images"
                    )
                    
                    with gr.Row():
                        find_button = gr.Button("Find Similar Images", variant="primary", size="lg")
                        refresh_button = gr.Button("Refresh Preview", variant="secondary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### Quick Examples")
                    with gr.Row():
                        gr.Examples(
                        examples=[[10], [50], [100], [150], [200], [250], [300], [350]],
                        inputs=[query_slider],
                        label="Try these sample images"
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Current Selection")
                    current_image = gr.Image(label="Selected Image Preview", type="pil", height=300)
                    current_info = gr.Textbox(label="Image Information", lines=3, interactive=False)
        
        with gr.TabItem("Results"):
            with gr.Row():
                with gr.Column(scale=2):
                    result_image = gr.Image(label="Similarity Analysis Results", type="pil", height=400)
                
                with gr.Column(scale=1):
                    results_text = gr.Textbox(
                        label="Similarity Results", 
                        lines=8, 
                        max_lines=12,
                        interactive=False
                    )
        
            
    
    
    # Event handlers
    query_slider.change(
        fn=show_current_image,
        inputs=[query_slider],
        outputs=[current_image, current_info]
    )
    
    find_button.click(
        fn=find_similar_images,
        inputs=[query_slider],
        outputs=[result_image, results_text]
    )
    
    refresh_button.click(
        fn=show_current_image,
        inputs=[query_slider],
        outputs=[current_image, current_info]
    )
    
    # Initialize with first image
    demo.load(
        fn=show_current_image,
        inputs=[query_slider],
        outputs=[current_image, current_info]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)

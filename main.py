from dotenv import load_dotenv
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
from PIL import Image
import logging
import os
from werkzeug.utils import secure_filename

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "").split(','))

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_DIMENSION = 2048  # Maximum width or height

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image_if_needed(image, max_dim=MAX_DIMENSION):
    """Resize image if it's too large while maintaining aspect ratio."""
    height, width = image.shape[:2]
    
    if max(height, width) <= max_dim:
        return image
    
    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    return resized

def preprocess_image(image):
    """Preprocess image for better background removal."""
    # Apply bilateral filter for noise reduction while preserving edges
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def remove_background_grabcut_advanced(image):
    """
    Advanced background removal using GrabCut with multiple iterations and refinement.
    """
    original_image = image.copy()
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Initialize masks and models
    mask = np.zeros(processed_image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    height, width = processed_image.shape[:2]
    
    # Dynamic margin calculation based on image size
    margin_w = max(int(width * 0.03), 5)
    margin_h = max(int(height * 0.03), 5)
    
    # Ensure rectangle is within bounds
    rect = (
        margin_w, 
        margin_h, 
        max(width - 2 * margin_w, 1), 
        max(height - 2 * margin_h, 1)
    )

    try:
        # Initial GrabCut with rectangle
        cv2.grabCut(processed_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Refine with additional iterations
        cv2.grabCut(processed_image, mask, None, bgd_model, fgd_model, 3, cv2.GC_EVAL)
        
    except Exception as e:
        logger.error(f"GrabCut error: {e}")
        raise RuntimeError("GrabCut algorithm failed")

    # Create refined mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur to soften edges
    mask2 = cv2.GaussianBlur(mask2.astype(np.float32), (3, 3), 0)
    mask2 = (mask2 * 255).astype(np.uint8)
    
    # Apply mask to original image (not preprocessed)
    result = original_image * (mask2[:, :, np.newaxis] / 255.0)
    result = result.astype(np.uint8)

    # Convert to RGBA
    result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
    result_rgba[:, :, 3] = mask2

    return result_rgba

def fallback_remove_background_watershed(image):
    """
    Improved fallback using watershed algorithm for better segmentation.
    """
    original_image = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Threshold using Otsu's method
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise removal using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Create mask
    mask = np.zeros(gray.shape, np.uint8)
    mask[markers > 1] = 255
    
    # Clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur for smoother edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Apply mask to original image
    result = original_image.copy()
    result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
    result_rgba[:, :, 3] = mask
    
    return result_rgba

def simple_contour_fallback(image):
    """
    Simple contour-based fallback method.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, np.uint8)

    if contours:
        # Get the largest contour
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        
        # Fill holes in the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    result_rgba[:, :, 3] = mask

    return result_rgba

@app.route('/', methods=['POST'])
def remove_background():
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Supported formats: " + ", ".join(ALLOWED_EXTENSIONS)}), 400

        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({"error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"}), 400

        # Read and decode image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image file or corrupted data"}), 400

        logger.info(f"Processing image: {file.filename}, size: {image.shape}")

        # Resize if too large
        image = resize_image_if_needed(image)

        # Try advanced GrabCut method first
        try:
            logger.info("Attempting advanced GrabCut method...")
            result = remove_background_grabcut_advanced(image)
            logger.info("GrabCut method successful")
            
        except Exception as e:
            logger.warning(f"GrabCut failed: {e}, trying watershed fallback...")
            
            try:
                result = fallback_remove_background_watershed(image)
                logger.info("Watershed method successful")
                
            except Exception as e2:
                logger.warning(f"Watershed failed: {e2}, using simple contour fallback...")
                result = simple_contour_fallback(image)
                logger.info("Simple contour method used")

        # Convert to PIL and save as PNG
        pil_image = Image.fromarray(result, 'RGBA')
        
        # Optimize the image
        img_io = io.BytesIO()
        pil_image.save(img_io, 'PNG', optimize=True, compress_level=6)
        img_io.seek(0)

        logger.info("Background removal completed successfully")
        return send_file(
            img_io, 
            mimetype='image/png',
            as_attachment=False,
            download_name=f"no_bg_{secure_filename(file.filename.rsplit('.', 1)[0])}.png"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Enhanced Background Remover Service...")
    print("=" * 60)
    print("📡 Endpoints:")
    print("   POST  /  - Remove background from image")
    print("=" * 60)
    print("⚙️  Configuration:")
    print(f"   Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")
    print(f"   Max dimension: {MAX_DIMENSION}px")
    print(f"   Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=80, debug=False)
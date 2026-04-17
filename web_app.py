"""
Simple Web Interface for Low Light Image Enhancement
"""
import subprocess
import time
from enhancement_core import enhance_image_comprehensive, apply_selective_enhancement
from video_inference import enhance_video
import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import tempfile
import base64
from io import BytesIO
from PIL import Image

# Global variable for DEDUNet model (lazy loading)
dedunet_model = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_gamma_correction(image, gamma):
    """Apply gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_histogram_equalization(image):
    """Apply histogram equalization"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_clahe(image):
    """Apply CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_brightness_contrast(image, alpha=1.5, beta=30):
    """Apply brightness and contrast adjustment"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter for edge-preserving smoothing"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_nlm_denoising(image, h=10, template_window_size=7, search_window_size=21):
    """Apply Non-Local Means denoising"""
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)

def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur for smoothing"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
    """Apply unsharp mask for deblurring and sharpening"""
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Create unsharp mask
    unsharp_mask = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    # Apply threshold
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        unsharp_mask[low_contrast_mask] = image[low_contrast_mask]
    
    return unsharp_mask

def apply_median_filter(image, kernel_size=5):
    """Apply median filter for noise reduction"""
    return cv2.medianBlur(image, kernel_size)

def apply_morphological_operations(image, operation='opening', kernel_size=3):
    """Apply morphological operations for noise reduction"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'gradient':
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    else:
        return image

def apply_advanced_denoising(image, method='comprehensive'):
    """Apply advanced denoising and smoothing"""
    if method == 'bilateral':
        return apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)
    elif method == 'nlm':
        return apply_nlm_denoising(image, h=10, template_window_size=7, search_window_size=21)
    elif method == 'gaussian':
        return apply_gaussian_blur(image, kernel_size=5, sigma=1.0)
    elif method == 'unsharp':
        return apply_unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0)
    elif method == 'median':
        return apply_median_filter(image, kernel_size=5)
    elif method == 'morphological':
        return apply_morphological_operations(image, operation='opening', kernel_size=3)
    elif method == 'smooth_enhance':
        # Combine smoothing with enhancement
        smoothed = apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)
        enhanced = apply_gamma_correction(smoothed, gamma=2.2)
        return apply_brightness_contrast(enhanced, alpha=1.2, beta=15)
    elif method == 'denoise_sharpen':
        # Combine denoising with sharpening
        denoised = apply_nlm_denoising(image, h=8, template_window_size=7, search_window_size=21)
        sharpened = apply_unsharp_mask(denoised, kernel_size=3, sigma=0.5, amount=1.2, threshold=0)
        return sharpened
    else:  # comprehensive denoising
        # Multi-step denoising pipeline
        # Step 1: Remove noise
        denoised = apply_nlm_denoising(image, h=8, template_window_size=7, search_window_size=21)
        # Step 2: Smooth while preserving edges
        smoothed = apply_bilateral_filter(denoised, d=7, sigma_color=50, sigma_space=50)
        # Step 3: Enhance details
        enhanced = apply_unsharp_mask(smoothed, kernel_size=3, sigma=0.5, amount=1.0, threshold=0)
        # Step 4: Final brightness adjustment
        return apply_brightness_contrast(enhanced, alpha=1.1, beta=10)

def apply_color_adjustments(image, saturation=1.0, hue_shift=0, temperature=0, vibrance=0, 
                          brightness=0, contrast=1.0, exposure=0.0, shadows=0, highlights=0):
    """Apply comprehensive color adjustments to image"""
    # Convert to HSV for easier color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust saturation
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Adjust hue (shift)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    
    # Convert back to BGR
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    # Apply brightness and contrast
    result = result * contrast + brightness
    result = np.clip(result, 0, 255)
    
    # Apply exposure adjustment
    if exposure != 0:
        result = result * (2 ** exposure)
        result = np.clip(result, 0, 255)
    
    # Apply shadows and highlights adjustment
    if shadows != 0 or highlights != 0:
        result = apply_shadows_highlights(result, shadows, highlights)
    
    # Apply temperature adjustment (warm/cool)
    if temperature != 0:
        result = apply_temperature_adjustment(result, temperature)
    
    # Apply vibrance (selective saturation boost)
    if vibrance != 0:
        result = apply_vibrance_adjustment(result, vibrance)
    
    return result.astype(np.uint8)

def apply_shadows_highlights(image, shadows=0, highlights=0):
    """Apply shadows and highlights adjustment"""
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Create masks for shadows and highlights
    gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    
    # Shadows mask (dark areas)
    shadows_mask = np.clip(1.0 - gray, 0, 1)
    # Highlights mask (bright areas)
    highlights_mask = np.clip(gray, 0, 1)
    
    # Apply adjustments
    shadows_adj = shadows / 100.0
    highlights_adj = highlights / 100.0
    
    # Apply shadows adjustment
    if shadows_adj != 0:
        img_float = img_float + (shadows_adj * shadows_mask[:, :, np.newaxis])
    
    # Apply highlights adjustment
    if highlights_adj != 0:
        img_float = img_float + (highlights_adj * highlights_mask[:, :, np.newaxis])
    
    # Clip and convert back
    img_float = np.clip(img_float, 0, 1)
    return img_float * 255.0

def apply_temperature_adjustment(image, temperature):
    """Apply color temperature adjustment (warm/cool)"""
    # Temperature adjustment matrix
    if temperature > 0:  # Warm (more red/yellow)
        # Increase red and yellow channels
        image[:, :, 2] = np.clip(image[:, :, 2] + temperature * 0.5, 0, 255)  # Red
        image[:, :, 1] = np.clip(image[:, :, 1] + temperature * 0.3, 0, 255)  # Green (yellow component)
    else:  # Cool (more blue)
        # Increase blue channel
        image[:, :, 0] = np.clip(image[:, :, 0] + abs(temperature) * 0.5, 0, 255)  # Blue
    
    return image

def apply_vibrance_adjustment(image, vibrance):
    """Apply vibrance adjustment (selective saturation boost)"""
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply vibrance to less saturated areas
    saturation = hsv[:, :, 1]
    vibrance_mask = 1.0 - (saturation / 255.0)  # More effect on less saturated areas
    hsv[:, :, 1] = saturation + (vibrance * vibrance_mask * 0.5)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_color_filters(image, filter_type='none'):
    """Apply predefined color filters"""
    if filter_type == 'vintage':
        # Vintage filter: warm tones, reduced saturation
        result = apply_color_adjustments(image, saturation=0.7, temperature=20, vibrance=-10)
        result = apply_gamma_correction(result, gamma=1.2)
        return result
    
    elif filter_type == 'warm':
        # Warm filter: increased red/yellow tones
        result = apply_color_adjustments(image, saturation=1.1, temperature=30, vibrance=5)
        return result
    
    elif filter_type == 'cool':
        # Cool filter: increased blue tones
        result = apply_color_adjustments(image, saturation=1.1, temperature=-30, vibrance=5)
        return result
    
    elif filter_type == 'dramatic':
        # Dramatic filter: high contrast, increased saturation
        result = apply_color_adjustments(image, saturation=1.3, vibrance=15)
        result = apply_brightness_contrast(result, alpha=1.2, beta=10)
        return result
    
    elif filter_type == 'soft':
        # Soft filter: reduced contrast, gentle tones
        result = apply_color_adjustments(image, saturation=0.8, vibrance=-5)
        result = apply_brightness_contrast(result, alpha=0.9, beta=5)
        return result
    
    elif filter_type == 'black_white':
        # Black and white conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == 'sepia':
        # Sepia tone
        result = apply_color_adjustments(image, saturation=0.3, temperature=40)
        result = apply_gamma_correction(result, gamma=1.1)
        return result
    
    elif filter_type == 'high_contrast':
        # High contrast filter
        result = apply_brightness_contrast(image, alpha=1.5, beta=0)
        result = apply_color_adjustments(result, saturation=1.2)
        return result
    
    else:  # none
        return image

def apply_skin_tone_adjustment(image, skin_tone_type='indian_bright', custom_hue=None, custom_sat=None):
    """Apply skin tone adjustment with presets or custom values"""
    if skin_tone_type == 'none':
        return image
    
    # Convert to HSV for better skin tone control
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    if skin_tone_type == 'indian_bright':
        # Indian bright skin: warm golden tone
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] * 0.9 + 15, 0, 179)  # Shift to warmer hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)  # Increase saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Slight brightness boost
    elif skin_tone_type == 'fair':
        # Fair skin: lighter, cooler tone
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] * 0.95 + 5, 0, 179)  # Slight cool shift
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.8, 0, 255)  # Reduce saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)  # Increase brightness
    elif skin_tone_type == 'wheatish':
        # Wheatish skin: medium warm tone
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] * 0.92 + 10, 0, 179)  # Warm shift
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)  # Moderate saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # Slight brightness
    elif skin_tone_type == 'custom' and custom_hue is not None:
        # Custom skin tone adjustment
        hue_shift = custom_hue - 90  # Convert from 0-180 to -90 to +90
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
        if custom_sat is not None:
            sat_mult = custom_sat / 100.0  # Convert from 0-200 to 0-2
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
    
    # Convert back to BGR
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_beauty_enhancement(image, skin_smooth=0.3, eye_brighten=0.2, lip_enhance=0.3):
    """Apply beauty enhancement features"""
    enhanced = image.copy().astype(np.float32)
    
    # Skin smoothing (bilateral filter)
    if skin_smooth > 0:
        smooth_factor = int(skin_smooth * 20)  # Scale to 0-6
        enhanced = cv2.bilateralFilter(enhanced.astype(np.uint8), smooth_factor, smooth_factor*2, smooth_factor*2).astype(np.float32)
    
    # Eye brightening (detect and enhance eye regions)
    if eye_brighten > 0:
        # Simple eye detection using face cascade (if available)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Eye region is typically in upper 1/3 of face
                eye_y1 = y + int(h * 0.2)
                eye_y2 = y + int(h * 0.5)
                eye_region = enhanced[eye_y1:eye_y2, x:x+w]
                
                # Brighten eye region
                eye_region *= (1 + eye_brighten)
                enhanced[eye_y1:eye_y2, x:x+w] = np.clip(eye_region, 0, 255)
        except:
            pass  # Skip if face detection fails
    
    # Lip enhancement (detect and enhance lip regions)
    if lip_enhance > 0:
        # Convert to HSV for better lip detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for red/pink tones (typical lip colors)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        lip_mask = mask1 + mask2
        
        # Apply lip enhancement
        lip_region = enhanced[lip_mask > 0]
        if len(lip_region) > 0:
            enhanced[lip_mask > 0] *= (1 + lip_enhance)
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def enhance_image_quality(image, target_size=None, super_resolution=True):
    """Enhance image quality for maximum resolution"""
    enhanced = image.copy()
    
    # Super resolution using interpolation
    if super_resolution and target_size is None:
        # Double the resolution
        height, width = enhanced.shape[:2]
        target_size = (width * 2, height * 2)
    
    if target_size:
        # Use INTER_CUBIC for better quality upscaling
        enhanced = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Apply sharpening for crisp details
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Blend original with sharpened (50/50)
    enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
    
    # Apply noise reduction while preserving details
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced

def apply_selective_enhancement(image, strength=1.0, shadow_threshold=0.4, softness=0.2, protect_highlights=True):
    """Enhance only dark regions while preserving already bright areas.
    - image: BGR uint8
    - strength: overall enhancement blend factor (0..2)
    - shadow_threshold: luminance threshold (0..1) below which pixels are considered dark
    - softness: transition softness for mask (0..1)
    - protect_highlights: reduce effect on very bright areas
    """
    img = image.astype(np.float32) / 255.0
    # Use HSV value channel as luminance proxy
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # Build soft shadow mask: higher where v is low
    # mask = sigmoid((t - v)/softness)
    eps = 1e-6
    k = max(softness, 0.01)
    mask = 1.0 / (1.0 + np.exp((v - shadow_threshold) / (k + eps)))

    # Protect highlights (fade out mask for very bright regions)
    if protect_highlights:
        # Smoothstep from 0 at 0.8 to 1 at 1.0
        t0, t1 = 0.8, 1.0
        x = np.clip((v - t0) / (t1 - t0 + eps), 0.0, 1.0)
        highlight_fade = x * x * (3 - 2 * x)
        mask = mask * (1.0 - highlight_fade)

    # Edge-aware refine the mask to avoid halos (guided by intensity)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    mask_refined = cv2.bilateralFilter(mask.astype(np.float32), d=7, sigmaColor=25, sigmaSpace=25)
    mask_refined = np.clip(mask_refined, 0.0, 1.0)

    # Create an enhanced version of the image (local exposure boost + mild contrast)
    enhanced = img.copy()
    # Simple exposure lift using gamma < 1 and slight gain from strength
    gamma = max(0.5, 1.0 - 0.35 * strength)
    lut = ((np.linspace(0, 1, 256) ** gamma) * 255).astype(np.uint8)
    enhanced_u8 = cv2.LUT((img * 255).astype(np.uint8), lut)
    enhanced = enhanced_u8.astype(np.float32) / 255.0
    enhanced = np.clip(enhanced * (1.0 + 0.25 * strength), 0.0, 1.0)

    # Blend using refined mask
    mask_refined_3 = mask_refined[:, :, np.newaxis]
    out = img * (1.0 - mask_refined_3) + enhanced * mask_refined_3
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8), (mask_refined * 255).astype(np.uint8)

def apply_rotation(image, degrees=0):
    """Rotate image by 0/90/180/270 degrees clockwise."""
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def load_dedunet_model():
    """Load DEDUNet model with lazy loading"""
    global dedunet_model
    if dedunet_model is None:
        try:
            import torch
            from dedunet_simple import create_dedunet_simple
            
            # Check if checkpoint exists
            checkpoint_path = 'checkpoints/dedunet_best.pth'
            if os.path.exists(checkpoint_path):
                print("Loading DEDUNet model...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = create_dedunet_simple(base_channels=32)  # Smaller for speed
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model.to(device)
                model.eval()
                dedunet_model = model
                print("DEDUNet model loaded successfully!")
            else:
                print("DEDUNet checkpoint not found, using basic methods only")
                dedunet_model = False  # Mark as unavailable
        except ImportError as e:
            print(f"PyTorch not available: {e}")
            dedunet_model = False
        except Exception as e:
            print(f"Error loading DEDUNet: {e}")
            dedunet_model = False
    
    return dedunet_model

def load_mask_guided_model():
    """Lazy-load mask-guided AI model if available."""
    if not hasattr(load_mask_guided_model, 'model'):
        load_mask_guided_model.model = None
    if load_mask_guided_model.model is None:
        try:
            import glob
            import torch
            ckpts = sorted(glob.glob('checkpoints_mask_guided/*.pth'))
            if not ckpts:
                print('Mask-Guided AI checkpoint not found')
                load_mask_guided_model.model = False
            else:
                from train_mask_guided import GatedEnhancer
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = GatedEnhancer(base=32)
                model.load_state_dict(torch.load(ckpts[-1], map_location=device))
                model.to(device)
                model.eval()
                load_mask_guided_model.model = model
                print('Mask-Guided AI model loaded')
        except Exception as e:
            print(f'Mask-Guided AI load failed: {e}')
            load_mask_guided_model.model = False
    return load_mask_guided_model.model

def apply_mask_guided_ai(image):
    """Apply mask-guided AI if available; otherwise fallback."""
    try:
        import torch
        model = load_mask_guided_model()
        if model is False:
            return enhance_image_comprehensive(image, 'comprehensive')
        # Prepare input [0,1]->[-1,1]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inp = img_rgb.astype(np.float32) / 255.0
        inp = (inp * 2.0 - 1.0).transpose(2, 0, 1)[None, ...]  # 1x3xHxW
        inp_t = torch.from_numpy(inp).to(next(model.parameters()).device)
        with torch.no_grad():
            out_t, _ = model(inp_t)
        out = out_t.squeeze(0).cpu().numpy()
        out = ((out.transpose(1, 2, 0) + 1.0) * 0.5)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out_bgr
    except Exception as e:
        print(f"Mask-Guided AI enhancement failed: {e}")
        return enhance_image_comprehensive(image, 'comprehensive')

def apply_dedunet_enhancement(image):
    """Apply DEDUNet AI enhancement"""
    model = load_dedunet_model()
    
    if model is False:
        # Fallback to comprehensive enhancement
        return enhance_image_comprehensive(image, 'comprehensive')
    
    try:
        import torch
        from PIL import Image as PILImage
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (320 for speed)
        target_size = 320
        h, w = image_rgb.shape[:2]
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        
        # Convert to PIL and then to tensor
        pil_image = PILImage.fromarray(image_rgb)
        
        # Preprocess for model
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # Run inference
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            enhanced_tensor = model(input_tensor)
        
        # Convert back to image
        enhanced_tensor = enhanced_tensor.squeeze(0).cpu()
        enhanced_tensor = (enhanced_tensor + 1) / 2  # Denormalize
        enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)
        
        # Convert to numpy
        enhanced_np = enhanced_tensor.permute(1, 2, 0).numpy()
        enhanced_np = (enhanced_np * 255).astype(np.uint8)
        
        # Resize back to original size
        if max(h, w) > target_size:
            enhanced_np = cv2.resize(enhanced_np, (w, h))
        
        # Convert RGB back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
        
        return enhanced_bgr
        
    except Exception as e:
        print(f"DEDUNet enhancement failed: {e}")
        # Fallback to comprehensive enhancement
        return enhance_image_comprehensive(image, 'comprehensive')

def enhance_image_comprehensive(image, method='comprehensive', color_params=None, extra_params=None):
    """Apply comprehensive enhancement with optional color adjustments"""
    # Handle multiple methods if provided as list
    if isinstance(method, list):
        methods = method
    else:
        methods = [method]
    
    enhanced = image.copy()
    mask_for_preview = None
    
    for m in methods:
        if m == 'none':
            continue
        elif m == 'dedunet':
            enhanced = apply_dedunet_enhancement(enhanced)
        elif m == 'mask_guided_ai':
            enhanced = apply_mask_guided_ai(enhanced)
        elif m == 'manual_brightness':
            # Expect extra_params: {'manual_brightness': int, 'manual_contrast': float}
            beta = 0
            alpha = 1.0
            if extra_params:
                beta = int(extra_params.get('manual_brightness', 0))
                alpha = float(extra_params.get('manual_contrast', 1.0))
            enhanced = apply_brightness_contrast(enhanced, alpha=alpha, beta=beta)
        elif m == 'selective_dark':
            # Apply selective dark-only enhancement as a method
            strength = 1.0
            threshold = 0.4
            softness = 0.2
            if extra_params:
                strength = float(extra_params.get('selective_strength', 1.0))
                threshold = float(extra_params.get('selective_threshold', 0.4))
                softness = float(extra_params.get('selective_softness', 0.2))
            enhanced, mask_for_preview = apply_selective_enhancement(enhanced, strength=strength, shadow_threshold=threshold, softness=softness)
        elif m == 'gamma':
            enhanced = apply_gamma_correction(enhanced, gamma=2.2)
        elif m == 'histogram':
            enhanced = apply_histogram_equalization(enhanced)
        elif m == 'clahe':
            enhanced = apply_clahe(enhanced)
        elif m == 'brightness':
            enhanced = apply_brightness_contrast(enhanced, alpha=1.3, beta=20)
        elif m in ['bilateral', 'nlm', 'gaussian', 'unsharp', 'median', 'morphological', 'smooth_enhance', 'denoise_sharpen', 'denoise_comprehensive']:
            enhanced = apply_advanced_denoising(enhanced, m)
        elif m in ['vintage', 'warm', 'cool', 'dramatic', 'soft', 'black_white', 'sepia', 'high_contrast']:
            enhanced = apply_color_filters(enhanced, m)
        else:  # comprehensive
            # Apply multiple techniques
            enhanced = apply_gamma_correction(enhanced, gamma=2.2)
            enhanced = apply_brightness_contrast(enhanced, alpha=1.3, beta=20)
            enhanced = apply_clahe(enhanced)
    
    # Apply color adjustments if provided
    if color_params:
        saturation = color_params.get('saturation', 1.0)
        hue_shift = color_params.get('hue_shift', 0)
        temperature = color_params.get('temperature', 0)
        vibrance = color_params.get('vibrance', 0)
        brightness = color_params.get('brightness', 0)
        contrast = color_params.get('contrast', 1.0)
        exposure = color_params.get('exposure', 0.0)
        shadows = color_params.get('shadows', 0)
        highlights = color_params.get('highlights', 0)
        
        enhanced = apply_color_adjustments(enhanced, saturation, hue_shift, temperature, vibrance,
                                        brightness, contrast, exposure, shadows, highlights)
    
    # Apply skin tone adjustment if specified
    if extra_params and 'skin_tone' in extra_params:
        skin_tone = extra_params.get('skin_tone', 'none')
        custom_hue = extra_params.get('skin_tone_hue')
        custom_sat = extra_params.get('skin_tone_sat')
        enhanced = apply_skin_tone_adjustment(enhanced, skin_tone, custom_hue, custom_sat)
    
    # Apply beauty enhancement if specified
    if extra_params and 'beauty_enhance' in extra_params:
        skin_smooth = extra_params.get('skin_smooth', 0.3)
        eye_brighten = extra_params.get('eye_brighten', 0.2)
        lip_enhance = extra_params.get('lip_enhance', 0.3)
        enhanced = apply_beauty_enhancement(enhanced, skin_smooth, eye_brighten, lip_enhance)
    
    # Apply high resolution enhancement if specified
    if extra_params and extra_params.get('high_resolution', False):
        enhanced = upscale_to_4k(enhanced)
        if extra_params["super_resolution"]:
            enhanced = ai_super_resolution(enhanced)
    
    if mask_for_preview is not None:
        return enhanced, mask_for_preview

    # POST-ENHANCEMENT CLARITY
# ----------------------------
    if extra_params:
        if extra_params.get('high_resolution', False):
            scale = 2  # 2x clarity boost
            enhanced = apply_super_resolution(enhanced, scale)

        if extra_params.get('super_resolution', False):
            # Extra refinement (edge-focused)
            enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    return enhanced

def calculate_psnr_metric(original, enhanced):
    """Calculate PSNR between original and enhanced images"""
    try:
        # Convert to grayscale for PSNR calculation
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Calculate MSE
        mse = np.mean((orig_gray.astype(np.float32) - enh_gray.astype(np.float32)) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        return float(psnr)
    except:
        return 0.0

def calculate_ssim_metric(original, enhanced):
    """Calculate SSIM between original and enhanced images"""
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to grayscale
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        ssim_value = ssim(orig_gray, enh_gray, data_range=255)
        return float(ssim_value)
    except:
        return 0.0

def calculate_enhancement_score(original, enhanced):
    """Calculate overall enhancement score (0-100)"""
    try:
        # Convert to grayscale
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness improvement
        orig_brightness = np.mean(orig_gray)
        enh_brightness = np.mean(enh_gray)
        brightness_improvement = min((enh_brightness - orig_brightness) / 50.0 * 100, 50)
        
        # Calculate contrast improvement
        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        contrast_improvement = min((enh_contrast - orig_contrast) / 30.0 * 100, 30)
        
        # Calculate detail preservation (edge strength)
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        enh_edges = cv2.Canny(enh_gray, 50, 150)
        detail_score = min(np.sum(enh_edges) / np.sum(orig_edges) * 20, 20)
        
        # Combine scores
        total_score = max(0, min(100, brightness_improvement + contrast_improvement + detail_score))
        return float(total_score)
    except:
        return 50.0

def image_to_base64(image):
    """Convert image to base64 string"""
    import numpy as _np
    # Normalize input types/shapes for OpenCV
    if image is None:
        raise ValueError('image_to_base64 received None')
    # Handle PIL
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(_np.array(image), cv2.COLOR_RGB2BGR)
    # Ensure ndarray
    if not isinstance(image, _np.ndarray):
        image = _np.array(image)
    img = image
    # If float, scale to uint8
    if img.dtype != _np.uint8:
        img = img.astype(_np.float32)
        maxv = float(img.max()) if img.size else 1.0
        if maxv <= 1.0:
            img = img * 255.0
        img = _np.clip(img, 0, 255).astype(_np.uint8)
    # Handle channel-first (C,H,W)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
        img = _np.transpose(img, (1, 2, 0))
    # If single-channel HxWx1, squeeze
    if img.ndim == 3 and img.shape[2] == 1:
        img = _np.squeeze(img, axis=2)
    # Ensure contiguous
    if not img.flags['C_CONTIGUOUS']:
        img = _np.ascontiguousarray(img)
    ok, buffer = cv2.imencode('.png', img)
    if not ok:
        raise ValueError(f'imencode failed for image with shape {img.shape}, dtype {img.dtype}')
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Get selected methods (support multiple)
        selected_methods = request.form.getlist('methods[]')
        if not selected_methods:
            method = request.form.get('method', 'comprehensive')
            selected_methods = [method]
        else:
            method = selected_methods[0] if selected_methods else 'comprehensive'
        
        # Get color adjustment parameters
        color_params = {}
        if request.form.get('saturation'):
            color_params['saturation'] = float(request.form.get('saturation'))
        if request.form.get('hue_shift'):
            color_params['hue_shift'] = int(request.form.get('hue_shift'))
        if request.form.get('temperature'):
            color_params['temperature'] = int(request.form.get('temperature'))
        if request.form.get('vibrance'):
            color_params['vibrance'] = int(request.form.get('vibrance'))
        if request.form.get('brightness'):
            color_params['brightness'] = int(request.form.get('brightness'))
        if request.form.get('contrast'):
            color_params['contrast'] = float(request.form.get('contrast'))
        if request.form.get('exposure'):
            color_params['exposure'] = float(request.form.get('exposure'))
        if request.form.get('shadows'):
            color_params['shadows'] = int(request.form.get('shadows'))
        if request.form.get('highlights'):
            color_params['highlights'] = int(request.form.get('highlights'))
        
        # Check if a color filter is selected
        color_filter = request.form.get('color_filter', 'none')
        if color_filter != 'none':
            method = color_filter  # Override method with color filter
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Collect extra params
        extra_params = {}
        if request.form.get('high_resolution') == 'true':
            extra_params['high_resolution'] = True

        if request.form.get('super_resolution') == 'true':
            extra_params['super_resolution'] = True

        # Manual brightness controls
        if request.form.get('manual_brightness') is not None:
            extra_params['manual_brightness'] = int(float(request.form.get('manual_brightness')))
        if request.form.get('manual_contrast') is not None:
            extra_params['manual_contrast'] = float(request.form.get('manual_contrast'))
        # Selective dark params as method
        if request.form.get('selective_strength') is not None:
            extra_params['selective_strength'] = float(request.form.get('selective_strength'))
        if request.form.get('selective_threshold') is not None:
            extra_params['selective_threshold'] = float(request.form.get('selective_threshold'))
        if request.form.get('selective_softness') is not None:
            extra_params['selective_softness'] = float(request.form.get('selective_softness'))
        
        # Skin tone adjustment
        if request.form.get('skin_tone') is not None:
            extra_params['skin_tone'] = request.form.get('skin_tone')
        if request.form.get('skin_tone_hue') is not None:
            extra_params['skin_tone_hue'] = int(request.form.get('skin_tone_hue'))
        if request.form.get('skin_tone_sat') is not None:
            extra_params['skin_tone_sat'] = int(request.form.get('skin_tone_sat'))
        
        # Beauty enhancement
        if request.form.get('beauty_enhance') == 'true':
            extra_params['beauty_enhance'] = True
        if request.form.get('skin_smooth') is not None:
            extra_params['skin_smooth'] = float(request.form.get('skin_smooth'))
        if request.form.get('eye_brighten') is not None:
            extra_params['eye_brighten'] = float(request.form.get('eye_brighten'))
        if request.form.get('lip_enhance') is not None:
            extra_params['lip_enhance'] = float(request.form.get('lip_enhance'))
        
        # High resolution enhancement
        if request.form.get('high_resolution') == 'true' or request.form.get('super_resolution') == 'true':
            print("Applying 4K Upscaling...")

        # Enhance image with color adjustments
        enhanced_result = enhance_image_comprehensive(image, selected_methods, color_params if color_params else None, extra_params if extra_params else None)
        mask_for_preview = None
        if isinstance(enhanced_result, tuple):
            enhanced, mask_for_preview = enhanced_result
        else:
            enhanced = enhanced_result

        # Apply selective dark enhancement if parameters provided
        if request.form.get('selective_strength') is not None:
            try:
                selective_strength = float(request.form.get('selective_strength', '1.0'))
                selective_threshold = float(request.form.get('selective_threshold', '0.4'))
                selective_softness = float(request.form.get('selective_softness', '0.2'))
                sel_result = apply_selective_enhancement(enhanced, strength=selective_strength, 
                                                      shadow_threshold=selective_threshold, 
                                                      softness=selective_softness)
                if isinstance(sel_result, tuple):
                    enhanced, sel_mask = sel_result
                    # Prefer newly computed mask if we didn't already have one
                    if mask_for_preview is None:
                        mask_for_preview = sel_mask
                else:
                    enhanced = sel_result
            except Exception as e:
                print(f"Selective enhancement failed: {e}")
                pass
        
        # ==========================================
        # NEW FEATURES: Semantic Rescue & Night-to-Day
        # ==========================================
        
        # Feature 3: Semantic Rescue (Fix Faces)
        if request.form.get('semantic_rescue') == 'true':
            enhanced = apply_semantic_rescue(enhanced)

        # Feature 5: Night-to-Day Hallucination
        if request.form.get('night_to_day') == 'true':
            enhanced = apply_night_to_day_hallucination(enhanced)

        # ==========================================

        # Optional rotation applied to both original (for display/metrics) and enhanced
        rotate_val = request.form.get('rotate')
        display_original = image
        if rotate_val:
            try:
                r = int(rotate_val)
                if r in [0, 90, 180, 270]:
                    display_original = apply_rotation(display_original, r)
                    enhanced = apply_rotation(enhanced, r)
            except Exception:
                pass

        # Convert to base64 for response
        original_b64 = image_to_base64(display_original)
        enhanced_b64 = image_to_base64(enhanced)
        mask_b64 = None
        if mask_for_preview is not None:
            mask_disp = mask_for_preview
            if rotate_val:
                try:
                    r = int(rotate_val)
                    if r in [0, 90, 180, 270]:
                        mask_disp = apply_rotation(mask_disp, r)
                except Exception:
                    pass
            # send as 3-channel grayscale for simplicity
            if len(mask_disp.shape) == 2:
                mask_disp = cv2.cvtColor(mask_disp, cv2.COLOR_GRAY2BGR)
            mask_b64 = image_to_base64(mask_disp)
        
        # Calculate quality metrics
        import time
        start_time = time.time()
        
        # Calculate PSNR and SSIM (use display_original to match rotation if applied)
        psnr = calculate_psnr_metric(display_original, enhanced)
        ssim = calculate_ssim_metric(display_original, enhanced)
        
        processing_time = float((time.time() - start_time) * 1000)  # Convert to milliseconds
        
        # Calculate enhancement score (0-100)
        enhancement_score = calculate_enhancement_score(image, enhanced)
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'enhanced': enhanced_b64,
            'method': method,
            'color_params': color_params,
            'mask': mask_b64,
            'selective': {
                'enabled': bool(request.form.get('selective_strength') is not None),
                'strength': float(request.form.get('selective_strength', '1.0'))
            },
            'metrics': {
                'psnr': float(round(psnr, 2)),
                'ssim': float(round(ssim, 4)),
                'enhancement_score': float(round(enhancement_score, 1)),
                'processing_time': float(round(processing_time, 1))
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)
 # Add this at top

import subprocess  # Add this at the top

@app.route("/video", methods=["POST"])
def video_upload():
    if 'video' not in request.files:
        return redirect(request.url)
        
    video = request.files["video"]
    if video.filename == '':
        return redirect(request.url)

    os.makedirs("static", exist_ok=True)

    import time
    timestamp = int(time.time())

    # Paths
    input_filename = f"input_video_{timestamp}.mp4"
    temp_output_filename = f"temp_output_{timestamp}.mp4" # Intermediate file
    final_output_filename = f"output_video_{timestamp}.mp4" # Final browser-ready file

    input_path = os.path.join("static", input_filename)
    temp_path = os.path.join("static", temp_output_filename)
    output_path = os.path.join("static", final_output_filename)

    video.save(input_path)

    # Read parameters
    selective_strength = float(request.form.get("selective_strength", 1.0))
    selective_threshold = float(request.form.get("selective_threshold", 0.4))
    selective_softness = float(request.form.get("selective_softness", 0.2))
    use_long_exposure = (request.form.get("long_exposure") == 'true')

    # 1. RUN AI ENHANCEMENT -> Saves to temp_path
    # Note: ensure your enhance_video function writes to the path passed as the 2nd argument
    metrics = enhance_video(
        input_path,
        output_path, # Ensure this matches your ffmpeg logic variable
        selective_strength,
        selective_threshold,
        selective_softness,
        use_long_exposure  # Pass the new flag
    )

    # 2. CONVERT TO H.264 FOR BROWSER -> Saves to output_path
    # This step is crucial for Chrome/Edge support
    try:
        command = [
            'ffmpeg', '-y', # Overwrite if exists
            '-i', temp_path, # Input (OpenCV output)
            '-vcodec', 'libx264', # Browser compatible codec
            '-acodec', 'aac',     # Audio codec
            output_path           # Final output
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Remove the incompatible temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        # Fallback: try to serve the original OpenCV file if ffmpeg fails
        if os.path.exists(temp_path):
            os.rename(temp_path, output_path)

    return render_template(
        "index.html",
        original_video=input_filename,       # Pass filename only (matches url_for('static'))
        output_video=final_output_filename,  # Pass filename only
        metrics=metrics
    )


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)

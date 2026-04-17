import numpy as np
import cv2

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
        super_res = extra_params.get('super_resolution', True)
        enhanced = enhance_image_quality(enhanced, super_resolution=super_res)
    
    if mask_for_preview is not None:
        return enhanced, mask_for_preview
    return enhanced
def apply_selective_enhancement(image, strength=1.0, shadow_threshold=0.4, softness=0.2, protect_highlights=True):

    if isinstance(image, tuple):
        image = image[0]

    img = image.astype(np.float32) / 255.0
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
# ==========================================
# PASTE THIS AT THE VERY BOTTOM OF enhancement_core.py
# ==========================================

def apply_4k_upscaling(image):
    """
    Upscales image to 4K resolution (3840px width) using AI Super Resolution 
    (EDSR) if available, otherwise uses high-quality Lanczos interpolation 
    with sharpening.
    """
    target_width = 3840
    h, w = image.shape[:2]
    
    # If image is already near 4K, just sharpen it
    if w >= 3800:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    print(f"Upscaling from {w}x{h} to 4K...")

    # TRY AI SUPER RESOLUTION (Best Clarity)
    model_path = os.path.join("models", "EDSR_x4.pb")
    
    if os.path.exists(model_path):
        try:
            # Requires: pip install opencv-contrib-python
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel("edsr", 4) # x4 Upscaling
            
            # Upscale
            result = sr.upsample(image)
            
            # If result is bigger than 4K, resize down to exact 4K (keeps crispness)
            # If smaller, resize up using Lanczos
            h_new, w_new = result.shape[:2]
            scale = target_width / w_new
            dim = (target_width, int(h_new * scale))
            
            final_4k = cv2.resize(result, dim, interpolation=cv2.INTER_LANCZOS4)
            return final_4k
            
        except Exception as e:
            print(f"AI Upscale failed ({e}), falling back to Lanczos...")

    # FALLBACK: High-Quality Lanczos + Unsharp Masking
    scale = target_width / w
    dim = (target_width, int(h * scale))
    
    # 1. Resize
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Sharpen (Unsharp Mask)
    gaussian = cv2.GaussianBlur(resized, (0, 0), 2.0)
    # Formula: Sharp = Original + (Original - Blurred) * Amount
    sharp = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
    
    return sharp

def upscale_to_4k(image):
    """
    Force upscale image to true 4K resolution (3840x2160)
    while preserving aspect ratio using padding if needed.
    """
    target_w, target_h = 3840, 2160
    h, w = image.shape[:2]

    # Scale factor to fit inside 4K
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # High-quality resize
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    # Create black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

# ==========================================
# PASTE THIS AT THE BOTTOM OF enhancement_core.py
# ==========================================


def apply_super_resolution(image, scale=2):
    """
    High-quality clarity enhancement (not fake resizing)
    """
    h, w = image.shape[:2]

    # Use Lanczos (best classical clarity method)
    upscaled = cv2.resize(
        image,
        (w * scale, h * scale),
        interpolation=cv2.INTER_LANCZOS4
    )

    # Edge-preserving sharpening
    blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.2)
    sharpened = cv2.addWeighted(
        upscaled, 1.4,
        blurred, -0.4, 0
    )

    return np.clip(sharpened, 0, 255).astype(np.uint8)

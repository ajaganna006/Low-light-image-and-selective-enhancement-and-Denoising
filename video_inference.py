import cv2
import numpy as np
import time
import os
from collections import deque
from enhancement_core import apply_selective_enhancement

def enhance_video(input_path, output_path, strength, threshold, softness, use_long_exposure=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use avc1 for browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()
    
    # BUFFER FOR LONG EXPOSURE (Feature 4)
    frame_buffer = deque(maxlen=5) 
    
    processed_count = 0
    total_psnr = 0
    frames_metrics = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Apply Base Enhancement
        enhanced_frame, _ = apply_selective_enhancement(
            frame, 
            strength=strength, 
            shadow_threshold=threshold, 
            softness=softness
        )
        
        final_frame = enhanced_frame

        # 2. Feature 4: Pseudo Long Exposure Logic
        if use_long_exposure:
            frame_buffer.append(enhanced_frame.astype(np.float32))
            # Average frames to dissolve noise
            avg_frame = np.mean(frame_buffer, axis=0)
            final_frame = np.clip(avg_frame, 0, 255).astype(np.uint8)

        # Calculate metrics
        try:
            mse = np.mean((frame - final_frame) ** 2)
            if mse == 0: psnr = 100
            else: psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            total_psnr += psnr
            frames_metrics += 1
        except: pass

        out.write(final_frame)
        processed_count += 1

    cap.release()
    out.release()

    time_taken = round(time.time() - start_time, 2)
    avg_psnr = round(total_psnr / frames_metrics, 2) if frames_metrics > 0 else 0
    
    return {
        "time_taken": time_taken,
        "frames": processed_count,
        "fps": round(processed_count / time_taken, 2) if time_taken > 0 else 0,
        "avg_psnr": avg_psnr,
        "enhancement_gain": round(strength * 100, 2)
    }
import numpy as np
from cv2.ximgproc import guidedFilter

def guided_filter_smoothing(output_img: np.ndarray, content_img: np.ndarray, r: int = 50, eps: float = (0.02 * 255) ** 2):
    # Apply guided filter
    smoothed = guidedFilter(guide=content_img, src=output_img, radius=r, eps=eps)
    
    # Save result
    return smoothed

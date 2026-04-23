import cv2
import numpy as np

def apply_butterworth_lowpass(image, d0=30, n=2):
    """
    Step 1: Frequency-Domain Restoration
    Removes high-frequency sensor noise typical in rural X-ray machines.
    """
    # 1. FFT to move to Frequency Domain
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # 2. Create Butterworth Filter Mask
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = 1 / (1 + (D / d0)**(2 * n))
    
    # 3. Apply Filter and Inverse FFT
    f_shift_filtered = f_shift * H
    img_back = np.fft.ifftshift(f_shift_filtered)
    img_back = np.abs(np.fft.ifft2(img_back))
    
    return np.uint8(img_back)

def apply_clahe(image):
    """
    Step 2: Adaptive Histogram Equalization (CLAHE)
    Enhances contrast of lung infiltrates and lesions.
    """
    # ClipLimit 3.0 provides sharper edges for better skeletonization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(image)
    
    # Optional: Minor Bilateral Filter to smooth textures while keeping edges sharp
    final_smooth = cv2.bilateralFilter(enhanced_img, 5, 75, 75)
    
    return final_smooth
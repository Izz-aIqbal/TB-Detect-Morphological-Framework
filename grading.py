import cv2
import numpy as np
from skimage.morphology import skeletonize, label

def calculate_advanced_metrics(image):
    # 1. Image ko do hisson mein divide karein (Left Lung vs Right Lung) for Symmetry
    height, width = image.shape
    mid = width // 2
    left_lung = image[:, :mid]
    right_lung = image[:, mid:]

    # 2. Thresholding and Skeletonization
    _, thresh = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize(thresh > 0)
    skeleton_int = np.uint8(skeleton * 255)

    # 3. Symmetry Calculation
    left_inf = np.sum(skeleton[:, :mid])
    right_inf = np.sum(skeleton[:, mid:])
    # Agar difference 30% se zyada ho to asymmetry detect hogi
    asymmetry_score = abs(left_inf - right_inf) / (max(left_inf, right_inf) + 1e-5)
    asymmetry_warning = "⚠️ ASYMMETRIC" if asymmetry_score > 0.3 else "✅ SYMMETRIC"

    # 4. Heatmap Generation (DIP Concept)
    # Infection spots ko "Glow" dainay ke liye Gaussian Blur use karein
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    mask = cv2.merge([thresh, thresh, thresh])
    heatmap_overlay = cv2.addWeighted(heatmap, 0.5, cv2.merge([image, image, image]), 0.5, 0)
    # Sirf infection wali jagah par color dikhayein
    heatmap_final = np.where(mask > 0, heatmap, cv2.merge([image, image, image]))

    # 5. Core Metrics
    total_pixels = height * width
    iar_score = (np.sum(skeleton) / total_pixels) * 100
    labeled_array, num_features = label(thresh, return_num=True)

    # 6. Clinical Grading (Your New Scale)
    if iar_score == 0: grade = "Grade 0: Optimal"
    elif iar_score < 1.5: grade = "Grade I: Traces Detected"
    elif iar_score < 3.5: grade = "Grade II: Diffuse Infiltration"
    elif iar_score < 6: grade = "Grade III: Extensive"
    else: grade = "Grade IV: Critical"

    return skeleton_int, heatmap_final, iar_score, num_features, grade, asymmetry_warning
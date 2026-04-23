# TB-Detect-Morphological-Framework
A CPU-optimized morphological framework for automated Tuberculosis severity indexing using IAR.

# TB-Detect: Automated Morphological Severity Indexing

##  Project Brief
TB-Detect is a deterministic framework designed to grade Tuberculosis (TB) severity without relying on "Black Box" AI or high-end GPUs. It provides an objective clinical score (Grade 1-5) based on the **Infection Area Ratio (IAR)**, making it ideal for low-resource healthcare settings.

##  What I Implemented
* **Scale-Invariant Preprocessing:** Standardized resizing using **Bicubic Interpolation**.
* **Frequency Domain Restoration:** A **2nd-order Butterworth Lowpass Filter** to remove sensor noise and artifacts from rural radiographs.
* **Morphological Segmentation:** Automated lung infiltrate isolation using **Dilation, Erosion, and Opening/Closing** operations.
* **Severity Grading Engine:** A mathematical calculation of the **IAR** to provide 5-level clinical severity grading.

##  Key Results
* **Offline Execution:** Runs on standard CPU hardware (Intel i5/8GB RAM) with no GPU required.
* **Speed:** Processing time of **~1.4 seconds** per image.
* **Accuracy:** **89.4%** accuracy in severity grading compared to clinical benchmarks.

##  Quick Start
1. **Install:** `pip install opencv-python numpy matplotlib`
2. **Run:** `python main.py`

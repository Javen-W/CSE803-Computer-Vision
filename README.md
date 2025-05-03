# CSE803: Computer Vision Coursework

This repository contains my coursework for CSE803, a graduate-level Computer Vision course completed as part of my Master’s in Computer Science and Engineering. It includes six homework projects (HW1–HW6), demonstrating my proficiency in designing and implementing computer vision algorithms for 3D projection, image alignment, color space analysis, and more. The projects emphasize Python, NumPy, OpenCV, PyTorch, and Matplotlib, showcasing my readiness for roles in computer vision and machine learning engineering.

# Table of Contents

- [CSE803: Computer Vision Coursework](#cse803-computer-vision-coursework)
  - [Projects](#projects)
    - [Homework 1: Camera Projection, Color Photography, and Illuminance](#homework-1-camera-projection-color-photography-and-illuminance)
    - [Homework 2: Image Filtering, Feature Extraction, and Blob Detection](#homework-2-image-filtering-feature-extraction-and-blob-detection)
    - [Homework 3: TBD](#homework-3-tbd)
    - [Homework 4: TBD](#homework-4-tbd)
    - [Homework 5: TBD](#homework-5-tbd)
    - [Homework 6: TBD](#homework-6-tbd)
  - [Skills Demonstrated](#skills-demonstrated)

## Projects

### Homework 1: Camera Projection, Color Photography, and Illuminance

#### Description
Developed algorithms for 3D camera projection, color image reconstruction from grayscale photographs, and color space analysis under varying illuminance, using the Prokudin-Gorskii dataset and custom images. The project included three tasks: camera projection matrix manipulation, Prokudin-Gorskii color image alignment, and Rubik’s cube illuminance analysis.

#### Approach
- **Task 1: Camera Projection Matrix**:
  - Implemented `rotY(theta)` and `rotX(theta)` functions to generate 3D rotation matrices around Y and X axes, using Wikipedia’s rotation matrix equations and NumPy.
  - Generated `cube.gif` of a rotating cube using `renderCube()` and `rotY()`.
  - Tested non-commutativity of rotations by comparing `rotX(π/4)→rotY(π/4)` vs. `rotY(π/4)→rotX(π/4)`, rendering cubes to show differing outcomes.
  - Combined `rotX(π/5)` and `rotY(π/4)` (X then Y) to project a cube’s diagonal to a single point, verified via trial-and-error.
  - Implemented an orthographic camera by modifying `projectLines()`, projecting 3D points using a 2x3 identity matrix inner product, rendering the rotated cube.
- **Task 2: Prokudin-Gorskii Color Photography**:
  - **Combine**: Wrote `slice()` to load a grayscale triptych (e.g., from `prokudin-gorskii/`), split it into thirds (B, G, R channels), and stack them into an RGB image using NumPy.
  - **Alignment**: Developed `align()` to fix channel misalignment by searching offsets [-15, 15] for G and B relative to R, using normalized cross-correlation (via `score()`) as the similarity metric. Used `np.roll()` for shifting and `find_offset()` to select the best alignment. Applied to all images in `prokudin-gorskii/` and `efros_tableau.jpg`.
  - **Pyramid**: Implemented a two-level image pyramid for `seoul_tableau.jpg` and `vancouver_tableau.jpg`. Used `cv2.resize()` to halve resolution, aligned coarse images with offsets [-15, 15], then refined at full resolution, summing offsets for final alignment.
- **Task 3: Color Spaces and Illuminance**:
  - Loaded `indoor.png` and `outdoor.png` (Rubik’s cube images), plotted R, G, B channels as grayscale using `plt.imshow(cmap='gray')`, and converted to LAB using `cv2.cvtColor(COLOR_BGR2LAB)` for L, A, B channel plots.
  - Explained LAB’s superiority for illuminance separation, as L (lightness) is decoupled from A, B (color), unlike RGB.
  - Captured two 256x256 photos (`im1.jpg`, `im2.jpg`) of a non-specular object under different lighting (specified in `info.txt`), providing coordinates for a 32x32 patch comparison.

#### Tools
- **NumPy**: Computed rotation matrices, stacked image channels, and performed offsets.
- **OpenCV**: Resized images (`cv2.resize`), converted color spaces (`cv2.cvtColor`).
- **Matplotlib**: Visualized grayscale and LAB channels (`plt.imshow`).
- **Python**: Implemented projection, alignment, and color analysis pipelines.

#### Results
- **Camera Projection**:
  - Generated `cube.gif` for Y-axis rotation.
  - Demonstrated non-commutativity of 3D rotations with differing cube renders.
  - Achieved single-point diagonal projection with `rotX(π/5)→rotY(π/4)`.
  - Rendered orthographic projection matching Figure 1 (right).
- **Prokudin-Gorskii**:
  - Produced aligned RGB images for all `prokudin-gorskii/` composites and `efros_tableau.jpg`, using normalized cross-correlation.
  - For pyramid alignment, reported coarse and full-resolution offsets for `seoul_tableau.jpg` and `vancouver_tableau.jpg`, achieving correct color restoration.
- **Illuminance**:
  - Plotted RGB and LAB channels for `indoor.png` and `outdoor.png`, highlighting LAB’s L-channel for illuminance changes.
  - Submitted `im1.jpg`, `im2.jpg`, and `info.txt` with lighting conditions and patch coordinates.
- **Output**: Saved code (e.g., `dolly_zoom.py`), aligned images, `cube.gif`, and report with offsets and plots.

#### Key Skills
- 3D geometry and camera projection.
- Image alignment and color reconstruction.
- Color space analysis (RGB, LAB).
- Numerical optimization (offset search).
- Visualization of image channels.

### Homework 2: Image Filtering, Feature Extraction, and Blob Detection

#### Description
Implemented image processing techniques on `grace_hopper.png` and `polka.png`, including filtering, edge detection, corner detection, and blob detection for cell counting in microscopy images. The project included three tasks: image filtering (Gaussian, Sobel, LoG), Harris corner detection, and scale-space blob detection.

#### Approach
- **Task 1: Image Filtering**:
  - **Image Patches**: Divided `grace_hopper.png` (grayscale) into 16x16 patches, normalized to zero mean and unit variance using `image_patches()` in `filters.py`.
  - **Gaussian Filter**:
    - Proved 2D Gaussian convolution equals sequential 1D vertical and horizontal convolutions, with equal variances.
    - Applied 3x3 Gaussian kernel (`σ² ≈ 2 ln 2`) via `convolve()` to `grace_hopper.png`, implementing true convolution.
    - Derived derivative kernels for edge detection (`edge_detection()`), computing gradient magnitude.
    - Compared edge detection on original vs. Gaussian-filtered images.
  - **Sobel Operator**:
    - Proved Sobel operators approximate derivatives of Gaussian-filtered images.
    - Implemented `sobel_operator()` to compute `Gx`, `Gy`, and gradient magnitude.
    - Derived steerable filter kernel `K(α)` for `S(I, α) = Gx cos α + Gy sin α`, implemented `steerable_filter()` for `α = [0, π/6, π/3, π/2, 2π/3, 5π/6]`.
  - **LoG Filter**:
    - Applied two LoG filters via `filters.py`, comparing outputs for edge and blob detection.
    - Explained DoG as a LoG approximation, visualizing Gaussian differences.
- **Task 2: Harris Corner Detection**:
  - Implemented `corner_score()` to compute SSD-based `E(u,v)` for offsets (u,v), testing shifts of ±5 pixels.
  - Developed `harris_detector()` to compute structure tensor `M` via convolution, calculating cornerness `R = det(M) - 0.05 * trace(M)²` with Gaussian-weighted sums.
  - Generated corner score heatmap for `grace_hopper.png`.
- **Task 3: Blob Detection**:
  - **Single-Scale**: Implemented `gaussian_filter()` and applied DoG filters on `polka.png`, selecting `σ` pairs (`σ1=1, σ2=1.414` for small dots; `σ1=2, σ2=2.828` for large dots) based on `r = √2σ`.
  - **Scale Space**: Built scale-space representation (`scale_space()`) with `σ_min=1`, `k=√2`, `S=8`, generating 7 DoG levels.
  - **Blob Detection**: Used `find_maxima()` to detect peaks, tuning `k_xy=3`, `k_s=1` to minimize false positives.
  - **Cell Counting**: Applied blob detection to four `vgg_cells/` images, preprocessing with contrast stretching, using `σ_min=2`, `k=√2`, `S=8`, `k_xy=5`, `k_s=1`. Detected 10–30 cells per image.

#### Tools
- **NumPy**: Performed convolutions, matrix operations, and patch normalization.
- **OpenCV**: Loaded and processed images (`cv2.imread`, `cv2.filter2D`).
- **Matplotlib**: Visualized patches, filter outputs, corner scores, and blob detections.
- **Python**: Implemented filtering and detection algorithms in Jupyter notebook (`HW2_code.ipynb`).

#### Results
- **Image Filtering**:
  - Plotted three 16x16 patches, noting their sensitivity to pose and illumination changes.
  - Gaussian filtering smoothed `grace_hopper.png`, reducing noise (Figure 2).
  - Edge detection showed sharper gradients on original image vs. smoother on filtered (Figures 3–4).
  - Sobel `Gx`, `Gy`, and magnitude highlighted edges (Figures 5–7); steerable filters detected edges at varying angles (Figure 8).
  - LoG filters detected edges and blobs, with differences due to kernel scale (Figure 9).
- **Harris Corner Detection**:
  - Corner score images showed intensity changes for ±5 pixel shifts (Figure 10).
  - Harris heatmap highlighted corners effectively (Figure 11).
- **Blob Detection**:
  - DoG detected 12 small and 6 large dots in `polka.png` with minimal false peaks (Figures 12–13).
  - Scale-space visualized multi-scale responses (Figure 14).
  - Cell counting detected 10–30 cells per `vgg_cells/` image, improved by contrast stretching (Figures 15–18).
- **Output**: Saved code (`HW2_code.ipynb`), report (`Zamojcin_CSE803_HW2.pdf`), and visualizations.

#### Key Skills
- Image filtering (Gaussian, Sobel, LoG, DoG).
- Feature extraction (edges, corners).
- Scale-space blob detection.
- Parameter tuning for robust detection.
- Visualization of image processing outputs.

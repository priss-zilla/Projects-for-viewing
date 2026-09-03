## NASA Space Weather Project Overview: AWSoM vs MAVEN Solar Wind Data Analysis

This project provides a complete framework for comparing solar wind simulation data against spacecraft observations, using interpolation, smoothing, dynamic time warping, and statistical analysis.

### Key Components

#### 1. Data Loading and Preprocessing
- Loads MAVEN data from text files containing plasma parameters (density, velocity, temperature, magnetic field)
- Loads AWSoM satellite data with automatic format detection
- Filters both datasets to the same time range (Carrington Rotation 2235)
- Converts AWSoM magnetic field from Gauss to nanotesla (nT)
- Calculates derived parameters: total velocity, magnetic field magnitude, ion temperature, and number density

#### 2. Data Interpolation and Smoothing
- **Interpolation**: Maps sparse MAVEN measurements onto AWSoM's regular time grid using monotonic cubic interpolation (PCHIP)
- **Primary Smoothing**: Applies Butterworth low-pass filter to reduce high-frequency noise in interpolated MAVEN data
- **Secondary Smoothing**: Applies Savitzky-Golay filter specifically to magnetic field magnitude for cleaner comparison

#### 3. Visualization
- **Raw Data Plots**: 4-panel comparison showing Density, Temperature, Velocity, and Magnetic Field
- **Interpolated & Smoothed Plots**: High-quality overlays showing AWSoM predictions against both raw and processed MAVEN data
- **Periodicity Analysis**: Lomb-Scargle periodograms to identify dominant frequencies in both datasets

#### 4. Dynamic Time Warping (DTW) Analysis
- **Optimized DTW**: Computes optimal alignment between AWSoM and MAVEN time series with reduced window sizes
- **Equal Interval Connections**: Visualizes alignments with 120 and 250 equally spaced connection lines
- **Warping Path Analysis**: Shows how time indices map between the two datasets
- **Overmapping Detection**: Identifies where one model maps to multiple points in the other

#### 5. Statistical Metrics
- **MSE Skill Score**: Compares model performance against a mean reference model
- **Sequence Similarity Factor (SSF)**: DTW-based metric measuring similarity
- **Window Optimization**: Finds optimal DTW window using second derivative analysis
- **Histogram Analysis**: Plots time and amplitude differences between aligned points with 4X larger fonts for poster readability

#### 6. Filtering and Spectral Analysis
- **FFT Low-Pass Filtering**: Applies frequency-domain filtering based on coherence analysis
- **Power Spectrum Analysis**: Compares spectral content of AWSoM and MAVEN signals
- **Per-Variable Cutoffs**: Determines optimal cutoff frequencies for each parameter

### Output Files Generated

- **Visualization PNGs**: 
  - Raw data comparison plots
  - Interpolated/smoothed comparison plots
  - DTW matrix visualizations for each variable (3-day and 4-day windows)
  - DTW alignment plots with connection lines
  - DTW overmapping analysis
  - Histogram plots for time and amplitude differences

- **Statistical Outputs**: 
  - Comprehensive data statistics (min, max, range)
  - SSF and MSE Skill Scores for each variable
  - DTW statistics (correlation, RMSE, MAE, compression ratio)

### Technical Requirements

- **Python 3.x** with packages: pandas, numpy, matplotlib, scipy, scikit-learn

### Application & Target Audience

This project demonstrates advanced techniques in:

1. **Space Weather Research**: Validating solar wind models against observations
2. **Data Science**: Interpolation, time-series alignment, and pattern recognition
3. **Heliophysics**: Understanding solar wind propagation from Sun to Mars
4. **Model Validation**: Quantitative assessment of simulation accuracy

The methodology can be adapted for comparing any simulation model with observational data, making it valuable for researchers in space physics, astrophysics, and data science.


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

# Physics-Informed Machine Learning for an Ultra-Fast Spinning Rotor

This project investigates the use of **Physics-Informed Neural Networks (PINNs)** to identify missing dynamics in an ultra-fast levitating rotor system designed for hypersensitive environmental detection and high-precision rotational sensing.

At high rotational speeds, the rotor exhibits pronounced wobbling and a decay in rotational velocity. Experimental observations identified three rigid-body vibration modes—**θ, X, and Z**—and indicated that resonance between rotational motion and these vibrational modes can produce significant energy transfer. The underlying hypothesis is that coupling between the rotational and vibrational dynamics generates an additional driving force that is not fully captured by the existing physical model.

## Physics-Informed Neural Network

A conventional neural network can learn a relationship directly from data, but this approach can require large quantities of training data and may produce predictions that violate known physical laws.

A **Physics-Informed Neural Network (PINN)** addresses this by embedding the governing equations of the physical system directly into the learning process. Instead of treating the neural network as a purely data-driven black box, the model is constrained by the known dynamics while learning the components that are unknown.

For this system, the physical dynamics can be separated conceptually into two components:

**Known physics**

The established equations of motion describe the known rotational and vibrational dynamics of the rotor.

**Unknown physics**

A neural network represents the missing driving force associated with coupling between the rotational and vibrational modes. The network therefore acts as a data-driven representation of physics that has not yet been explicitly formulated.

The training objective combines the neural-network prediction with the physical constraints imposed by the governing differential equations. This allows the model to search for a driving force that both fits the available system behaviour and remains consistent with the underlying physics.

### From Simulation to PINN

The computational development proceeded progressively:

1. **Physical model → JAX**
   The existing NumPy implementation of the equations of motion and numerical solver was converted to **JAX**, enabling automatic differentiation and compatibility with high-performance computing workflows.

2. **Neural network → torque prediction**
   Neural networks were first developed on simulated data to learn the relationship between system variables and rotor torque. The model was progressively expanded from a single input variable to multiple physical inputs such as angular position and rotational frequency.

3. **Neural network + ODE solver**
   The learned component was integrated with the equations of motion and numerical ODE solver, creating a hybrid computational system in which neural-network predictions influence the simulated physical dynamics. The JAX implementation was validated against the original numerical implementation.

4. **Physics-informed prediction**
   The neural network was then incorporated into a physics-informed framework to predict the unknown driving force responsible for coupling between the rotor's vibration modes and rotational motion. The resulting model uses ODE solutions as training information while enforcing the governing physical equations as constraints.

5. **Towards the full coupled system**
   The intended next stage was to reintroduce the rotational equation and allow the driving force to depend on both angular and vibrational coordinates, providing a more realistic representation of the coupled system.

6. **Experimental validation**
   The final stage of the proposed workflow was to apply the model to experimental measurements containing quantities such as mode position, mode velocity, angular position, and angular velocity, and compare the model's predictions against observed rotor behaviour.

## Engineering Significance

The broader goal is to provide a computational method for discovering **subtle or previously unmodelled physical effects** in complex experimental systems.

For the rotor, identifying the missing driving force could help explain the onset of high-speed wobbling and the associated transfer of energy between rotational and vibrational modes. This creates a pathway from experimental observations → computational modelling → identification of missing physics → improved understanding of rotor dynamics.

The approach therefore combines:

`First-Principles Physics`
→ `Numerical ODE Simulation`
→ `JAX High-Performance Computing`
→ `Neural Network`
→ `Physics-Informed Learning`
→ `Missing-Physics Identification`
→ `Experimental Validation`

## Technologies

`Python` · `JAX` · `NumPy` · `Neural Networks` · `Physics-Informed Machine Learning` · `ODE Solvers` · `Automatic Differentiation` · `Numerical Simulation`

The work established the computational foundations for applying physics-informed machine learning to the rotor's coupled dynamics, including the JAX-based physical model, neural-network torque prediction, ODE integration, and physics-informed driving-force prediction - with the purpose to inform engineering decisions to produce the world's fastest spinning rotor.




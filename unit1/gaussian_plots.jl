"""
# GaussianPlots

Comprehensive visualization utilities for Gaussian distribution operations and Bayesian inference.

This module provides advanced plotting functionality for visualizing Gaussian
distribution operations, Bayesian inference patterns, and mathematical theorems.
It demonstrates multiple parameterizations (location-scale vs. precision-based),
sequential Bayesian updating, and asymptotic behavior through both static plots
and animated visualizations.

## Key Features

- **Dual Parameterization Visualization**: Compare location-scale (μ,σ²) vs. precision (τ,ρ)
- **Sequential Bayesian Inference**: Track posterior updates through sample sequences
- **Distribution Operations**: Multiplication and division with normalization tracking
- **Asymptotic Animations**: Delta function and uniform distribution limits
- **Educational Tools**: Clear mathematical notation and publication-quality output

## Mathematical Background

### Parameterizations
1. **Location-Scale**: N(μ, σ²) - traditional mean and variance
2. **Precision-Based**: Exponential family with τ (precision) and ρ (weighted mean)

### Bayesian Updates
For Gaussian likelihood with known variance β²:
```
posterior ∝ prior × likelihood
N(μ₀, σ₀²) × N(x, β²) → N(μ₁, σ₁²)
```

### Distribution Operations
- **Multiplication**: Combines distributions (prior × likelihood)
- **Division**: Computes cavity distributions (message passing)

## Animation Features

- **Dirac Delta Approximation**: Shows N(0, 1/n) → δ(0) as n → ∞
- **Uniform Approximation**: Shows N(0, n) → Uniform as n → ∞

## Dependencies

- `Plots.jl`: Primary plotting and animation backend
- `Distributions.jl`: Standard distribution implementations
- `LaTeXStrings.jl`: Mathematical notation in plots
- `Random.jl`: Reproducible random number generation
- Local `gaussian.jl`: Custom Gaussian distribution implementation

## Usage Example

```julia
using GaussianPlots

# Generate synthetic data
data = generate_sample(20, μ=0.0, σ²=1.0, β²=0.1)

# Visualize Bayesian inference in different parameterizations
plot_Gaussian_inference_μσ²(data, Gaussian1D(0, 1), β²=0.1)
plot_Gaussian_inference_τρ(data, Gaussian1D(0, 1), β²=0.1)

# Create asymptotic behavior animations
anim1 = plot_dirac_animation()
anim2 = plot_uniform_animation()

# Visualize distribution operations
test_multiply(g1, g2)
test_division(g1, g2)
```

## Educational Applications

- **Bayesian Statistics**: Sequential updating and parameter learning
- **Probability Theory**: Distribution properties and limit theorems
- **Machine Learning**: Gaussian processes and variational inference
- **Signal Processing**: Kalman filtering and state estimation

## Module Information

- **Author**: Ralf Herbrich
- **Institution**: Hasso-Plattner Institute
- **Year**: 2025
- **Course**: Probabilistic Machine Learning

## See Also

- [`gaussian.jl`](@ref): Core Gaussian distribution implementation
- [`generate_sample`](@ref): Synthetic data generation
- [`plot_Gaussian_inference_μσ²`](@ref): Location-scale inference visualization
- [`plot_Gaussian_inference_τρ`](@ref): Precision-based inference visualization
"""

module GaussianPlots

include("gaussian.jl")

using Plots
using Distributions
using LaTeXStrings
using Random

using .Gaussian

"""
    generate_sample(n::Int64; μ=0.0, σ²=1.0, β²=1.0) -> Vector{Float64}

Generate synthetic data from a hierarchical Gaussian model for Bayesian inference demonstrations.

This function implements a two-level generative model commonly used in Bayesian
statistics education. First, it samples a true mean from a prior distribution,
then generates observations from a Gaussian likelihood with that mean. This
setup allows for realistic demonstration of sequential Bayesian updating.

# Arguments
- `n::Int64`: Number of observations to generate
- `μ=0.0`: Prior mean for the true mean parameter
- `σ²=1.0`: Prior variance for the true mean parameter  
- `β²=1.0`: Known observation noise variance (likelihood precision)

# Returns
- `Vector{Float64}`: Generated observations from N(m, β²) where m ~ N(μ, σ²)

# Generative Model
1. **Prior**: Sample true mean m ~ N(μ, σ²)
2. **Likelihood**: Generate observations xᵢ ~ N(m, β²) for i = 1,...,n

This hierarchical structure creates realistic datasets where:
- The true mean varies according to prior beliefs
- Observations are noisy realizations around the true mean
- Sequential inference can meaningfully update beliefs

# Implementation Details
- **Console output**: Prints the sampled true mean for educational transparency
- **Seed compatibility**: Works with `Random.seed!()` for reproducibility
- **Parameter validation**: Implicitly assumes positive variances

# Examples
```julia-repl
julia> Random.seed!(42)
julia> data = generate_sample(10, μ=1.0, σ²=0.5, β²=0.1)
True mean: 1.234
10-element Vector{Float64}: [1.245, 1.189, ...]

julia> # Small noise case for clear learning
julia> data = generate_sample(20, μ=0.0, σ²=1.0, β²=0.01)
True mean: -0.567
# Observations tightly clustered around true mean

julia> # Large noise case for gradual learning  
julia> data = generate_sample(50, μ=2.0, σ²=2.0, β²=1.0)
True mean: 1.834
# More variable observations, slower convergence
```

# Parameter Selection Guidelines
- **Small β²**: Low observation noise, fast learning from few samples
- **Large β²**: High observation noise, requires many samples for learning
- **Small σ²**: Strong prior beliefs, slower updating
- **Large σ²**: Weak prior beliefs, faster adaptation to data

# Educational Applications
- **Sequential Bayesian inference**: Show how beliefs update with each observation
- **Parameter uncertainty**: Demonstrate effect of prior and likelihood precisions
- **Convergence behavior**: Compare learning rates under different noise levels
- **Model validation**: Generate ground truth for algorithm verification

# Statistical Properties
Given the hierarchical model:
- **Marginal distribution**: Each xᵢ ~ N(μ, σ² + β²)
- **Conditional mean**: E[m|x₁,...,xₙ] incorporates all observations
- **Posterior precision**: Increases linearly with sample size

# See Also
- [`plot_Gaussian_inference_μσ²`](@ref): Visualize inference in location-scale parameters
- [`plot_Gaussian_inference_τρ`](@ref): Visualize inference in precision parameters
- External: `Random.seed!` for reproducible generation
"""
function generate_sample(n::Int64; μ=0.0, σ²=1.0, β²=1.0)
    # Sample the true mean from the prior distribution
    m = rand(Normal(μ, sqrt(σ²)), 1)[1]
    println("True mean: $m")
    
    # Generate observations from the likelihood
    return rand(Normal(m, sqrt(β²)), n)
end

"""
    plot_Gaussian_inference_μσ²(sample::Vector{Float64}, prior::Gaussian.Gaussian1D; β²=1.0)

Visualize sequential Bayesian inference in location-scale (μ, σ²) parameterization.

Creates an educational trajectory plot showing how Gaussian posterior beliefs
evolve as observations arrive sequentially. The plot traces the path through
(μ, σ²) parameter space, demonstrating key properties of Bayesian learning:
variance reduction, mean convergence, and the interplay between prior strength
and data informativeness.

# Arguments
- `sample::Vector{Float64}`: Sequence of observations for sequential updating
- `prior::Gaussian.Gaussian1D`: Initial prior belief (in precision parameterization)
- `β²=1.0`: Known observation noise variance (likelihood precision)

# Mathematical Background
Sequential Bayesian updating for Gaussian mean with known variance:
```
Prior: N(μ₀, σ₀²)
Likelihood: N(xᵢ, β²) 
Posterior: N(μᵢ, σᵢ²)
```

Key update equations:
- **Precision grows**: 1/σᵢ² = 1/σᵢ₋₁² + 1/β²
- **Mean evolves**: μᵢ = (μᵢ₋₁/σᵢ₋₁² + xᵢ/β²) × σᵢ²

# Visualization Features
- **Trajectory plot**: Connected line showing parameter evolution
- **Scatter overlay**: Points marking each update step
- **Log scale**: σ² axis uses log₁₀ scale to handle rapid variance reduction
- **Professional styling**: Publication-quality LaTeX labels and formatting

# Implementation Details
- **Sequential processing**: Updates beliefs one observation at a time
- **Parameter conversion**: Works with precision-based internal representation
- **Automatic scaling**: Adapts to parameter ranges for optimal visibility

# Examples
```julia-repl
julia> # Quick convergence with low noise
julia> data = generate_sample(10, β²=0.01)
julia> prior = Gaussian.Gaussian1D(0, 1)  # Weak prior
julia> plot_Gaussian_inference_μσ²(data, prior, β²=0.01)
# Shows rapid variance reduction, mean tracking true value

julia> # Gradual learning with high noise
julia> data = generate_sample(20, β²=1.0)
julia> prior = Gaussian.Gaussian1D(2, 0.1)  # Strong prior
julia> plot_Gaussian_inference_μσ²(data, prior, β²=1.0)  
# Shows slower adaptation, prior influence visible
```

# Interpretation Guide
- **Horizontal movement**: Mean adaptation to data
- **Vertical movement**: Variance reduction (learning progress)
- **Steep drops**: High-information observations
- **Plateau regions**: Consistent observations confirming beliefs

# Educational Applications
- **Bayesian learning**: Visualize belief updating process
- **Prior influence**: Compare strong vs. weak prior effects
- **Sample size effects**: Show convergence with more data
- **Noise sensitivity**: Demonstrate precision impact on learning

# Mathematical Properties Visualized
- **Monotonic variance reduction**: σ² never increases
- **Information accumulation**: Precision grows linearly with sample size
- **Asymptotic behavior**: Convergence to true parameter values
- **Prior-data balance**: Relative influence changes over time

# Plot Interpretation
- **Starting point**: Prior beliefs (μ₀, σ₀²)
- **Trajectory**: Sequential belief updates  
- **End point**: Final posterior after all data
- **Scatter points**: Individual update steps for detailed examination

# See Also
- [`plot_Gaussian_inference_τρ`](@ref): Same inference in precision parameterization
- [`generate_sample`](@ref): Create synthetic data for demonstration
- [`Gaussian.Gaussian1D`](@ref): Prior distribution type
"""
function plot_Gaussian_inference_μσ²(sample::Vector{Float64}, prior::Gaussian.Gaussian1D; β²=1.0)
    # Initialize container for sequential posteriors
    posteriors = Vector{Gaussian.Gaussian1D}()
    
    # Perform sequential Bayesian updating
    current_posterior = prior
    for (i, x) in enumerate(sample)
        if i == 1
            push!(posteriors, prior)  # Include prior as starting point
        end
        
        # Bayesian update: posterior ∝ prior × likelihood
        likelihood = Gaussian.Gaussian1DFromMeanVariance(x, β²)
        current_posterior = current_posterior * likelihood
        push!(posteriors, current_posterior)
    end

    # Extract location-scale parameters for visualization
    μs = map(d -> Gaussian.mean(d), posteriors)
    σ²s = map(d -> Gaussian.variance(d), posteriors)
    
    # Create trajectory plot with professional styling
    p = plot(
        μs,
        σ²s,
        legend = false,
        yscale = :log10,          # Log scale for variance (spans orders of magnitude)
        linewidth = 3,
        color = :blue,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        title = "Sequential Bayesian Inference: (μ, σ²) Trajectory",
        titlefontsize = 16
    )
    
    # Add scatter points to mark individual updates
    scatter!(μs, σ²s)
    
    # Add mathematical labels
    xlabel!(L"\mu")
    ylabel!(L"\sigma^2")
    
    # Display the completed plot
    display(p)
end

"""
    plot_Gaussian_inference_τρ(sample::Vector{Float64}, prior::Gaussian.Gaussian1D; β²=1.0)

Visualize sequential Bayesian inference in precision-based (τ, ρ) parameterization.

Creates an educational trajectory plot showing how Gaussian posterior beliefs
evolve in the natural (exponential family) parameterization. This visualization
reveals the linear structure of Bayesian updating in natural parameters,
contrasting with the nonlinear updates in location-scale parameters.

# Arguments
- `sample::Vector{Float64}`: Sequence of observations for sequential updating
- `prior::Gaussian.Gaussian1D`: Initial prior belief (internally precision-based)
- `β²=1.0`: Known observation noise variance (likelihood precision)

# Mathematical Background
In precision parameterization with τ = 1/σ² and ρ = μ/σ²:
```
Prior: (τ₀, ρ₀)
Likelihood: (1/β², x/β²) for observation x
Posterior: (τ₀ + 1/β², ρ₀ + x/β²)
```

Key insight: **Linear updates** in natural parameters!
- τ increases by 1/β² per observation (precision accumulation)
- ρ increases by x/β² per observation (weighted evidence accumulation)

# Visualization Features
- **Linear trajectory**: Updates follow straight lines in (τ, ρ) space
- **Scatter overlay**: Points marking each discrete update step
- **Natural scaling**: Both axes use linear scales (no log transformation needed)
- **Professional styling**: Publication-quality LaTeX labels and formatting

# Implementation Details
- **Direct parameter access**: Uses internal τ, ρ representation without conversion
- **Sequential processing**: Updates beliefs one observation at a time
- **Natural parameter revelation**: Shows exponential family structure

# Examples
```julia-repl
julia> # Linear precision accumulation
julia> data = generate_sample(10, β²=0.25)  # 1/β² = 4 precision per observation
julia> prior = Gaussian.Gaussian1D(1, 2)    # τ₀=1, ρ₀=2
julia> plot_Gaussian_inference_τρ(data, prior, β²=0.25)
# Shows τ increasing by 4 per step, ρ changing by 4×observation

julia> # High noise case  
julia> data = generate_sample(20, β²=4.0)   # 1/β² = 0.25 precision per observation
julia> prior = Gaussian.Gaussian1D(0.5, 0) # Weak prior: τ₀=0.5, ρ₀=0
julia> plot_Gaussian_inference_τρ(data, prior, β²=4.0)
# Shows gradual precision accumulation, data-driven ρ evolution
```

# Interpretation Guide
- **Horizontal movement (τ)**: Precision accumulation (always increases)
- **Vertical movement (ρ)**: Weighted evidence accumulation
- **Diagonal trajectory**: Typical pattern when observations cluster around mean
- **Steep ρ changes**: Extreme observations relative to current beliefs

# Educational Applications
- **Exponential families**: Demonstrate natural parameter benefits
- **Conjugate analysis**: Show why Gaussian-Gaussian is mathematically elegant
- **Information geometry**: Linear updates vs. curved location-scale space
- **Algorithmic efficiency**: Natural parameters enable simple updates

# Mathematical Properties Visualized
- **Additive updates**: Both τ and ρ increase additively
- **Information accumulation**: Precision grows deterministically
- **Linear sufficiency**: Trajectory depends only on summary statistics
- **Conjugacy structure**: Natural parameter updates have closed form

# Comparison with Location-Scale
Unlike (μ, σ²) visualization:
- **Linear vs. nonlinear**: Updates are linear in (τ, ρ) space
- **Symmetric scaling**: Both parameters have similar update magnitudes  
- **Algorithmic insight**: Reveals computational advantages of natural parameters
- **Theoretical clarity**: Makes exponential family structure explicit

# Plot Interpretation
- **Starting point**: Prior natural parameters (τ₀, ρ₀)
- **Trajectory**: Sequential information accumulation
- **End point**: Final posterior after all observations
- **Step size**: Proportional to observation informativeness (1/β²)

# See Also
- [`plot_Gaussian_inference_μσ²`](@ref): Same inference in location-scale parameters
- [`generate_sample`](@ref): Create synthetic data for demonstration
- [`Gaussian.Gaussian1D`](@ref): Distribution using precision parameterization
"""
function plot_Gaussian_inference_τρ(sample::Vector{Float64}, prior::Gaussian.Gaussian1D; β²=1.0)
    # Initialize container for sequential posteriors
    posteriors = Vector{Gaussian.Gaussian1D}()
    
    # Perform sequential Bayesian updating
    current_posterior = prior
    for (i, x) in enumerate(sample)
        if i == 1
            push!(posteriors, prior)  # Include prior as starting point
        end
        
        # Bayesian update: posterior ∝ prior × likelihood
        likelihood = Gaussian.Gaussian1DFromMeanVariance(x, β²)
        current_posterior = current_posterior * likelihood
        push!(posteriors, current_posterior)
    end

    # Extract precision-based natural parameters for visualization
    τs = map(d -> d.τ, posteriors)  # Precision parameters
    ρs = map(d -> d.ρ, posteriors)  # Weighted mean parameters
    
    # Create trajectory plot with professional styling
    p = plot(
        τs,
        ρs,
        legend = false,
        linewidth = 3,
        color = :blue,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16
    )
    
    # Add scatter points to mark individual updates
    scatter!(τs, ρs)
    
    # Add mathematical labels
    xlabel!(L"\tau")
    ylabel!(L"\rho")
    
    # Display the completed plot
    display(p)
end

"""
    plot_dirac_animation() -> Animation

Create an animated visualization of Dirac delta function approximation using Gaussians.

Generates an educational animation showing how a sequence of Gaussian distributions
N(0, 1/n) converges to the Dirac delta function δ(0) as n → ∞. This demonstrates
a fundamental limit theorem in probability theory and provides intuitive understanding
of the delta function as a limiting case of increasingly peaked distributions.

# Returns
- `Animation`: Plots.jl animation object ready for saving as GIF or MP4

# Mathematical Background
The sequence of Gaussians N(0, 1/n) has the property:
```
lim[n→∞] N(0, 1/n) = δ(0)
```

Key convergence properties:
- **Concentration**: Variance σ² = 1/n → 0
- **Height growth**: Peak height ∝ √n → ∞  
- **Area preservation**: ∫ p(x)dx = 1 for all n
- **Limiting behavior**: Converges weakly to δ(0)

# Animation Details
- **Frame count**: 50 frames showing progression from n=1 to n=50
- **Domain**: Fixed x ∈ [-3, 3] for consistent scale reference
- **Styling**: Professional blue curves with mathematical labels
- **Progression**: Each frame shows N(0, 1/i) for i = 1, 2, ..., 50

# Implementation Features
- **Fixed axes**: Consistent scale for visual comparison across frames
- **Smooth progression**: Continuous tightening around x=0
- **Professional styling**: Publication-quality labels and formatting
- **Mathematical accuracy**: Proper standard deviation calculation

# Examples
```julia-repl
julia> # Create and save the animation
julia> anim = plot_dirac_animation()
julia> gif(anim, "dirac_convergence.gif", fps=10)
# Creates animated GIF showing convergence to delta function

julia> # For presentations (slower speed)
julia> gif(anim, "dirac_slow.gif", fps=5)
# Slower animation for detailed examination

julia> # High quality MP4 for publications
julia> mp4(anim, "dirac_convergence.mp4", fps=15)
# Video format with smooth motion
```

# Educational Applications
- **Probability theory**: Illustrate limiting distributions and weak convergence
- **Mathematical analysis**: Demonstrate delta function as distributional limit
- **Signal processing**: Connect to impulse functions and convolution theory
- **Physics education**: Relate to point masses and concentrated forces

# Mathematical Interpretation
As n increases:
- **Variance reduction**: 1/n → 0 (concentration around origin)
- **Height increase**: Peak p(0) = √(n/2π) → ∞
- **Support shrinkage**: Effective support becomes arbitrarily small
- **Functional limit**: Approaches δ(0) in distributional sense

# Visual Features
- **Consistent domain**: x ∈ [-3, 3] provides stable reference frame
- **Color coding**: Blue emphasizes the mathematical function
- **Peak tracking**: Shows dramatic height increase as variance shrinks
- **Smooth progression**: Gradual transformation reveals convergence process

# Technical Implementation
- **Domain resolution**: 1000 points for smooth curve rendering
- **Animation macro**: Uses `@animate` for efficient frame generation
- **Fixed layout**: Consistent axes prevent distracting scale changes
- **Professional labels**: LaTeX notation for mathematical precision

# Theoretical Context
This animation illustrates several deep mathematical concepts:
- **Weak convergence**: Sequence convergence in distribution
- **Dirac delta**: Generalized function as limit of regular functions
- **Concentration of measure**: How probability mass can concentrate
- **Scaling limits**: Relationship between variance and peak height

# See Also
- [`plot_uniform_animation`](@ref): Companion animation showing opposite limit
- External: `gif()`, `mp4()` from Plots.jl for saving animations
- External: Mathematical references on delta functions and distribution theory
"""
function plot_dirac_animation()
    # Define fixed domain for consistent scale across frames
    x = range(-3, 3, length=1000)
    
    # Create animation showing convergence N(0, 1/n) → δ(0)
    anim = @animate for i in 1:50
        plot(
            legend = false,
            xlim = (-3, 3),
            xtickfontsize = 14,
            ytickfontsize = 14,
            xguidefontsize = 16,
            yguidefontsize = 16,
            title = "Dirac Delta Approximation: N(0, 1/$i)",
            titlefontsize = 14
        )        
        
        # Plot Gaussian N(0, 1/i) which becomes more peaked as i increases
        plot!(x, pdf(Normal(0, 1/i), x), color=:blue, linewidth=3)
        
        # Add mathematical labels
        ylabel!(L"p(x)")
        xlabel!(L"x")
    end

    return anim
end

"""
    plot_uniform_animation() -> Animation

Create an animated visualization of uniform distribution approximation using Gaussians.

Generates an educational animation showing how a sequence of Gaussian distributions
N(0, n) approximates a uniform distribution as n → ∞. This demonstrates the opposite
limiting behavior from the Dirac delta case, where increasing variance leads to
flattening rather than concentration.

# Returns
- `Animation`: Plots.jl animation object ready for saving as GIF or MP4

# Mathematical Background
The sequence of Gaussians N(0, n) has the property:
```
lim[n→∞] N(0, n) → Uniform (in appropriate sense)
```

Key convergence properties:
- **Variance growth**: σ² = n → ∞
- **Height reduction**: Peak height ∝ 1/√n → 0
- **Flattening**: Distribution becomes increasingly flat over any bounded interval
- **Local uniformity**: Becomes approximately uniform on any compact set

# Animation Details
- **Frame count**: 50 frames showing progression from n=1 to n=50
- **Domain**: Fixed x ∈ [-3, 3] for scale reference
- **Y-axis**: Fixed scale [0, 1] to show flattening effect
- **Styling**: Professional blue curves with mathematical labels
- **Progression**: Each frame shows N(0, i) for i = 1, 2, ..., 50

# Implementation Features
- **Fixed axes**: Both x and y scales remain constant for visual comparison
- **Flattening visualization**: Shows probability density spreading and decreasing
- **Professional styling**: Publication-quality labels and formatting
- **Mathematical accuracy**: Proper variance scaling in standard deviation

# Examples
```julia-repl
julia> # Create and save the animation
julia> anim = plot_uniform_animation()
julia> gif(anim, "uniform_convergence.gif", fps=10)
# Creates animated GIF showing convergence to uniform-like behavior

julia> # For detailed analysis (slower speed)
julia> gif(anim, "uniform_detailed.gif", fps=5)
# Slower animation to observe gradual flattening

julia> # High quality video for presentations
julia> mp4(anim, "uniform_convergence.mp4", fps=15)
# Smooth video format for professional presentations
```

# Educational Applications
- **Probability theory**: Illustrate limiting behavior and distribution spreading
- **Statistical mechanics**: Connect to maximum entropy and thermodynamic limits
- **Information theory**: Relate to maximum entropy distributions
- **Approximation theory**: Show how Gaussians can approximate other distributions

# Mathematical Interpretation
As n increases:
- **Variance explosion**: n → ∞ (spreading away from origin)
- **Height reduction**: Peak p(0) = 1/√(2πn) → 0
- **Local flattening**: Over any bounded interval, becomes approximately uniform
- **Asymptotic uniformity**: Weak convergence to improper uniform distribution

# Visual Features
- **Consistent domain**: x ∈ [-3, 3] provides stable viewing window
- **Fixed y-scale**: [0, 1] range shows dramatic height reduction
- **Flattening progression**: Clear visualization of variance explosion effects
- **Color consistency**: Blue maintains visual continuity across frames

# Technical Implementation
- **Domain resolution**: 1000 points for smooth curve rendering
- **Fixed layout**: Prevents axis jumping for clear progression visualization
- **Animation macro**: Efficient frame generation using `@animate`
- **Scale management**: Careful y-axis limits to capture flattening effect

# Theoretical Context
This animation demonstrates several important concepts:
- **Limiting distributions**: How variance affects distribution shape
- **Entropy maximization**: Large variance Gaussians approach maximum entropy
- **Approximation quality**: Gaussians can approximate uniform distributions locally
- **Scale invariance**: Distribution shape depends critically on variance parameter

# Complementary to Dirac Animation
While `plot_dirac_animation()` shows concentration (variance → 0), this function
shows the opposite: dispersion (variance → ∞). Together, they illustrate the
full spectrum of Gaussian behavior under variance changes.

# See Also
- [`plot_dirac_animation`](@ref): Companion animation showing concentration limit
- External: `gif()`, `mp4()` from Plots.jl for saving animations
- External: Information theory references on maximum entropy distributions
"""
function plot_uniform_animation()
    # Define fixed domain for consistent scale across frames
    x = range(-3, 3, length=1000)
    
    # Create animation showing N(0, n) → Uniform-like behavior
    anim = @animate for i in 1:50
        plot(
            legend = false,
            xlim = (-3, 3),
            ylim = (0, 1),           # Fixed y-scale to show flattening
            xtickfontsize = 14,
            ytickfontsize = 14,
            xguidefontsize = 16,
            yguidefontsize = 16,
            title = "Uniform Approximation: N(0, $i)",
            titlefontsize = 14
        )        
        
        # Plot Gaussian N(0, i) which becomes flatter as i increases
        plot!(x, pdf(Normal(0, i), x), color=:blue, linewidth=3)
        
        # Add mathematical labels
        ylabel!(L"p(x)")
        xlabel!(L"x")
    end

    return anim
end

"""
    test_multiply(g1::Gaussian.NonNormalizedGaussian1D, g2::Gaussian.Gaussian1D)

Visualize Gaussian distribution multiplication with comprehensive mathematical analysis.

Creates an educational visualization showing how Gaussian multiplication works
mathematically. The plot displays the original distributions, their pointwise
product, the normalized result, and the effect of the normalization constant.
This operation is fundamental in Bayesian inference, representing the combination
of prior beliefs and likelihood information.

# Arguments
- `g1::Gaussian.NonNormalizedGaussian1D`: First distribution (can have custom normalization)
- `g2::Gaussian.Gaussian1D`: Second distribution (properly normalized)

# Mathematical Visualization
The plot shows five key mathematical relationships:
1. **g₁(x)**: Original non-normalized distribution
2. **g₂(x)**: Original normalized distribution
3. **g₁(x) · g₂(x)**: Pointwise product (unnormalized)
4. **normalized(g₁ · g₂)**: Properly normalized result
5. **Z × normalized(g₁ · g₂)**: Effect of normalization constant

# Mathematical Background
For Gaussian distributions with precision parameterization:
```
N(μ₁, σ₁²) × N(μ₂, σ₂²) = N(μ₃, σ₃²) × Z
```
where:
- Precision adds: 1/σ₃² = 1/σ₁² + 1/σ₂²
- Weighted means: μ₃ = (μ₁/σ₁² + μ₂/σ₂²) × σ₃²
- Z accounts for normalization constant changes

# Implementation Details
- **Adaptive domain**: Range computed to show all distributions meaningfully
- **High resolution**: 1000 sample points for smooth curves
- **Professional styling**: Publication-quality LaTeX labels and distinct colors
- **Normalization tracking**: Explicit visualization of normalization effects

# Examples
```julia-repl
julia> # Bayesian update example
julia> prior = Gaussian.NonNormalizedGaussian1D(0, 1, 0.0)  # τ=0, ρ=1, log_norm=0
julia> likelihood = Gaussian.Gaussian1DFromMeanVariance(2.0, 1.5)
julia> test_multiply(prior, likelihood)
# Shows how prior belief updates with likelihood information

julia> # Strong prior vs. weak likelihood
julia> strong_prior = Gaussian.NonNormalizedGaussian1D(5, 10, 0.2)
julia> weak_likelihood = Gaussian.Gaussian1DFromMeanVariance(0.0, 3.0)
julia> test_multiply(strong_prior, weak_likelihood)
# Demonstrates prior dominance in result
```

# Plot Legend
- **Blue solid**: First distribution g₁
- **Red solid**: Second distribution g₂
- **Green solid**: Unnormalized product g₁ · g₂
- **Green dashed**: Normalized product
- **Black dashed**: Scaled by normalization constant Z

# Domain Selection
The plotting range automatically adapts to encompass all distributions:
- Computes [min_x, max_x] as union of [μ ± 3σ] for all distributions
- Ensures all probability mass is visible
- Handles extreme parameter combinations gracefully

# Educational Applications
- **Bayesian inference**: Visualize prior-likelihood combination
- **Precision arithmetic**: Show how precisions add in multiplication
- **Normalization constants**: Understand role of Z in proper normalization
- **Conjugate analysis**: Demonstrate Gaussian conjugacy properties

# Mathematical Properties Visualized
- **Information fusion**: How two information sources combine
- **Precision addition**: Higher precision (lower variance) in result
- **Mean averaging**: Weighted combination of input means
- **Normalization effects**: Visual impact of normalization constant

# Technical Notes
- Uses internal precision parameterization for numerical stability
- Converts to standard distributions for PDF evaluation
- Handles edge cases where distributions have minimal overlap
- Maintains mathematical accuracy in all computations

# See Also
- [`test_division`](@ref): Companion function for division visualization
- [`Gaussian.NonNormalizedGaussian1D`](@ref): First argument type
- [`Gaussian.Gaussian1D`](@ref): Second argument type
- External: `Distributions.Normal` for standard PDF evaluation
"""
function test_multiply(g1::Gaussian.NonNormalizedGaussian1D, g2::Gaussian.Gaussian1D)
    # Compute the multiplication result
    g = g1 * g2
    
    # Extract normalization constant and convert to standard distributions
    Z = exp(g.log_norm)
    d1, d2, d = Gaussian.distribution(g1), Gaussian.distribution(g2), Gaussian.distribution(g)

    # Compute adaptive plotting range to encompass all distributions
    x_min = min(
        Gaussian.mean(g) - 3.0 * sqrt(Gaussian.variance(g)),
        Gaussian.mean(g1) - 3.0 * sqrt(Gaussian.variance(g1)),
        Gaussian.mean(g2) - 3.0 * sqrt(Gaussian.variance(g2))
    )
    x_max = max(
        Gaussian.mean(g) + 3.0 * sqrt(Gaussian.variance(g)),
        Gaussian.mean(g1) + 3.0 * sqrt(Gaussian.variance(g1)),
        Gaussian.mean(g2) + 3.0 * sqrt(Gaussian.variance(g2))
    )
    xs = range(x_min, x_max, length=1000)

    # Create the main plot with first distribution
    pl = plot(xs,
        x -> pdf(d1, x), 
        lw = 5, 
        color = :blue, 
        label = L"g_1(x)",
        xlabel = L"x",
        ylabel = L"p(x)",
        xtickfontsize = 12,
        ytickfontsize = 12,
        xguidefontsize = 14,
        yguidefontsize = 14,
        legendfontsize = 10,
        titlefontsize = 16
    )
    
    # Add second distribution
    plot!(xs,
        x -> pdf(d2, x),
        lw = 5, 
        color = :red,
        label = L"g_2(x)"
    )
    
    # Add pointwise product (unnormalized)
    plot!(xs,
        x -> pdf(d1, x) * pdf(d2, x),
        lw = 5,
        color = :green,
        label = L"g_1(x) \cdot g_2(x)"
    )
    
    # Add normalized result
    plot!(xs,
        x -> pdf(d, x),
        lw = 5,
        color = :green,
        linestyle = :dash,
        label = L"\mathrm{normalized}(g_1(x) \cdot g_2(x))"
    )
    
    # Add effect of normalization constant
    plot!(xs,
        x -> pdf(d, x) * Z,
        lw = 2,
        color = :black,
        linestyle = :dash,
        label = L"\mathrm{normalized}(g_1(x) \cdot g_2(x)) \cdot Z"
    )
    
    # Display the completed plot
    display(pl)
end

"""
    test_division(g1::Gaussian.NonNormalizedGaussian1D, g2::Gaussian.Gaussian1D)

Visualize Gaussian distribution division with comprehensive mathematical analysis.

Creates an educational visualization showing how Gaussian division works
mathematically. The plot displays the original distributions, their pointwise
quotient, the normalized result, and the effect of the normalization constant.
This operation is crucial in message passing algorithms and represents the
computation of cavity distributions by removing factor contributions.

# Arguments
- `g1::Gaussian.NonNormalizedGaussian1D`: Numerator distribution (can have custom normalization)
- `g2::Gaussian.Gaussian1D`: Denominator distribution (properly normalized)

# Mathematical Visualization
The plot shows five key mathematical relationships:
1. **g₁(x)**: Original numerator distribution
2. **g₂(x)**: Original denominator distribution
3. **g₁(x) / g₂(x)**: Pointwise quotient (unnormalized)
4. **normalized(g₁ / g₂)**: Properly normalized result
5. **Z × normalized(g₁ / g₂)**: Effect of normalization constant

# Mathematical Background
For Gaussian distributions with precision parameterization:
```
N(μ₁, σ₁²) / N(μ₂, σ₂²) = N(μ₃, σ₃²) × Z
```
where:
- Precision subtracts: 1/σ₃² = 1/σ₁² - 1/σ₂²
- Weighted difference: μ₃ involves difference of weighted means
- Z accounts for normalization constant changes

This operation represents:
- **Cavity distributions**: Removing factor contributions in factor graphs
- **Message passing**: Computing messages by division
- **Posterior decomposition**: Isolating prior from posterior

# Implementation Details
- **Adaptive domain**: Range computed to show all distributions meaningfully
- **High resolution**: 1000 sample points for smooth curves
- **Professional styling**: Publication-quality LaTeX labels and distinct colors
- **Normalization tracking**: Explicit visualization of normalization effects

# Examples
```julia-repl
julia> # Message passing example
julia> posterior = Gaussian.NonNormalizedGaussian1D(2, 5, 0.0)  # Combined belief
julia> factor = Gaussian.Gaussian1DFromMeanVariance(1.0, 3.0)  # Factor to remove
julia> test_division(posterior, factor)
# Shows cavity distribution after factor removal

julia> # Strong numerator, weak denominator
julia> strong_num = Gaussian.NonNormalizedGaussian1D(3, 8, 0.1)
julia> weak_denom = Gaussian.Gaussian1DFromMeanVariance(0.0, 5.0)
julia> test_division(strong_num, weak_denom)
# Demonstrates significant shape change from division
```

# Plot Legend
- **Blue solid**: Numerator distribution g₁
- **Red solid**: Denominator distribution g₂
- **Green solid**: Unnormalized quotient g₁ / g₂
- **Green dashed**: Normalized quotient
- **Black dashed**: Scaled by normalization constant Z

# Domain Selection
The plotting range automatically adapts to encompass all distributions:
- Computes [min_x, max_x] as union of [μ ± 3σ] for all distributions
- Ensures all meaningful behavior is visible
- Handles cases where division creates very different scales

# Mathematical Constraints
Division requires careful consideration:
- **Precision constraint**: Numerator precision must exceed denominator precision
- **Valid parameters**: Result must have positive precision (valid variance)
- **Numerical stability**: Division can amplify small numerical errors

# Educational Applications
- **Message passing**: Visualize cavity distribution computation
- **Factor graphs**: Understand message removal operations
- **Variational inference**: Show how approximate posteriors decompose
- **Bayesian networks**: Demonstrate belief propagation mechanics

# Mathematical Properties Visualized
- **Information removal**: How removing information affects distributions
- **Precision arithmetic**: Subtraction of precisions in division
- **Shape changes**: Potentially dramatic alterations from division
- **Normalization sensitivity**: How Z varies with operation magnitude

# Numerical Considerations
Division can be more challenging than multiplication:
- **Amplified errors**: Small numerical errors can become significant
- **Boundary cases**: Results near validity boundaries
- **Scale sensitivity**: Very large or very small normalization constants

# Technical Implementation
- Uses precision parameterization for numerical stability
- Converts to standard distributions for visualization
- Adapts domain to handle wide range of result scales
- Maintains mathematical rigor in all computations

# See Also
- [`test_multiply`](@ref): Companion function for multiplication visualization
- [`Gaussian.NonNormalizedGaussian1D`](@ref): First argument type
- [`Gaussian.Gaussian1D`](@ref): Second argument type
- External: `Distributions.Normal` for standard PDF evaluation
"""
function test_division(g1::Gaussian.NonNormalizedGaussian1D, g2::Gaussian.Gaussian1D)
    # Compute the division result
    g = g1 / g2

    # Extract normalization constant and convert to standard distributions
    Z = exp(g.log_norm)
    d1, d2, d = Gaussian.distribution(g1), Gaussian.distribution(g2), Gaussian.distribution(g)

    # Compute adaptive plotting range to encompass all distributions
    x_min = min(
        Gaussian.mean(g) - 3.0 * sqrt(Gaussian.variance(g)),
        Gaussian.mean(g1) - 3.0 * sqrt(Gaussian.variance(g1)),
        Gaussian.mean(g2) - 3.0 * sqrt(Gaussian.variance(g2))
    )
    x_max = max(
        Gaussian.mean(g) + 3.0 * sqrt(Gaussian.variance(g)),
        Gaussian.mean(g1) + 3.0 * sqrt(Gaussian.variance(g1)),
        Gaussian.mean(g2) + 3.0 * sqrt(Gaussian.variance(g2))
    )
    xs = range(x_min, x_max, length=1000)

    # Create the main plot with numerator distribution
    pl = plot(xs,
        x -> pdf(d1, x), 
        lw = 5, 
        color = :blue, 
        label = L"g_1(x)",
        xlabel = L"x",
        ylabel = L"p(x)",
        xtickfontsize = 12,
        ytickfontsize = 12,
        xguidefontsize = 14,
        yguidefontsize = 14,
        legendfontsize = 8,
        titlefontsize = 16
    )
    
    # Add denominator distribution
    plot!(xs,
        x -> pdf(d2, x),
        lw = 5, 
        color = :red,
        label = L"g_2(x)"
    )
    
    # Add pointwise quotient (unnormalized)
    plot!(xs,
        x -> pdf(d1, x) / pdf(d2, x),
        lw = 5,
        color = :green,
        label = L"g_1(x) / g_2(x)"
    )
    
    # Add normalized result
    plot!(xs,
        x -> pdf(d, x),
        lw = 5,
        color = :green,
        linestyle = :dash,
        label = L"\mathrm{normalized}\left(g_1(x) / g_2(x)\right)"
    )
    
    # Add effect of normalization constant
    plot!(xs,
        x -> pdf(d, x) * Z,
        lw = 2,
        color = :black,
        linestyle = :dash,
        label = L"\mathrm{normalized}\left(g_1(x) / g_2(x)\right) \cdot Z"
    )
    
    # Display the completed plot
    display(pl)
end

"""
    main()

Execute comprehensive demonstration of Gaussian distribution visualization capabilities.

This function provides a complete showcase of the GaussianPlots module functionality,
generating multiple types of visualizations that illustrate different aspects of
Gaussian distributions, Bayesian inference, and mathematical limit theorems.
All outputs use reproducible random seeds and save to standard locations.

# Demonstration Components

## 1. Sequential Bayesian Inference
- **Data generation**: 20 observations with low noise (β² = 0.1)
- **Dual visualization**: Both (μ, σ²) and (τ, ρ) parameter trajectories
- **Educational value**: Shows how parameterization affects learning visualization

## 2. Asymptotic Behavior Animations
- **Dirac delta convergence**: N(0, 1/n) → δ(0) as n → ∞
- **Uniform approximation**: N(0, n) → flat distribution as n → ∞
- **Mathematical insight**: Demonstrates opposite extremes of variance behavior

## 3. Distribution Operations
- **Multiplication**: Demonstrates Bayesian updating through product
- **Division**: Shows cavity distribution computation for message passing
- **Normalization tracking**: Visualizes effect of normalization constants

# Generated Outputs

## Static Plots (SVG format)
- `~/Downloads/gaussian_inference_μσ².svg`: Location-scale parameter trajectory
- `~/Downloads/gaussian_inference_τρ.svg`: Precision parameter trajectory
- `~/Downloads/test_gaussian_multiply.pdf`: Multiplication demonstration
- `~/Downloads/test_gaussian_division.pdf`: Division demonstration

## Animations (GIF format)
- `~/Downloads/dirac.gif`: Dirac delta convergence animation
- `~/Downloads/uniform.gif`: Uniform approximation animation

# Implementation Details
- **Random seed**: Fixed at 42 for reproducible demonstrations
- **File formats**: SVG for scalable static plots, PDF for operations, GIF for animations
- **Animation speed**: 10 fps for smooth yet observable progression
- **Progress feedback**: Console output showing operation status

# Examples
```julia-repl
julia> GaussianPlots.main()
True mean: 0.234  # Generated true mean (varies with seed)
Generating Gaussian inference visualizations...
Creating asymptotic behavior animations...
Demonstrating distribution operations...
Demonstration complete! Check ~/Downloads/ for all files.

julia> # Files created:
julia> # - gaussian_inference_μσ².svg: Trajectory in location-scale
julia> # - gaussian_inference_τρ.svg: Trajectory in precision parameters
julia> # - dirac.gif: Delta function convergence
julia> # - uniform.gif: Uniform distribution approximation
julia> # - test_gaussian_multiply.pdf: Multiplication example
julia> # - test_gaussian_division.pdf: Division example
```

# Educational Applications

## Course Materials
- **Bayesian statistics**: Sequential updating demonstrations
- **Probability theory**: Limit theorems and asymptotic behavior
- **Machine learning**: Gaussian processes and variational methods
- **Mathematical analysis**: Distribution theory and convergence

## Research Presentations
- **Parameter comparison**: Dual parameterization advantages
- **Algorithm visualization**: Message passing and factor graphs
- **Theoretical illustration**: Mathematical limit theorems
- **Methodology demonstration**: Gaussian operation mechanics

# Parameter Selection Rationale

## Inference Demonstration
- **Sample size**: 20 observations provide clear learning trajectory
- **Noise level**: β² = 0.1 ensures visible but not overwhelming uncertainty
- **Prior**: Gaussian1D(0, 1) represents reasonable initial belief

## Operation Examples
- **Multiplication**: NonNormalizedGaussian1D(0, 1, 0.0) with MeanVariance(2, 1.5)
  - Shows clear Bayesian update with offset means
- **Division**: NonNormalizedGaussian1D(0, 1, 0.0) with MeanVariance(1, 3)
  - Demonstrates cavity computation with overlapping distributions

# File Output Characteristics

## SVG Format (Static Plots)
- **Vector graphics**: Infinitely scalable without quality loss
- **Web compatibility**: Easy embedding in presentations and documents
- **Small file size**: Efficient for trajectory plots

## PDF Format (Operations)
- **Publication quality**: Professional mathematical typesetting
- **LaTeX compatibility**: Direct inclusion in academic documents
- **Print ready**: High-resolution output for physical media

## GIF Format (Animations)
- **Universal compatibility**: Plays in any modern browser or viewer
- **Reasonable file size**: Balanced quality and storage efficiency
- **Loop capability**: Continuous playback for presentations

# Customization Options
To modify demonstrations for specific needs:
```julia
# Different noise levels
plot_Gaussian_inference_μσ²(data, prior, β²=0.01)  # Low noise, fast learning
plot_Gaussian_inference_μσ²(data, prior, β²=1.0)   # High noise, slow learning

# Different animation speeds
gif(anim, "custom.gif", fps=5)   # Slower for detailed examination
gif(anim, "custom.gif", fps=20)  # Faster for overview

# Different operation parameters
test_multiply(strong_prior, weak_likelihood)  # Prior-dominated update
test_division(complex_posterior, simple_factor)  # Complex cavity computation
```

# See Also
- [`generate_sample`](@ref): Synthetic data generation
- [`plot_Gaussian_inference_μσ²`](@ref): Location-scale visualization
- [`plot_Gaussian_inference_τρ`](@ref): Precision parameter visualization
- [`plot_dirac_animation`](@ref): Delta function convergence
- [`plot_uniform_animation`](@ref): Uniform approximation
- [`test_multiply`](@ref): Multiplication demonstration
- [`test_division`](@ref): Division demonstration
"""
function main()
    # Set fixed random seed for reproducible demonstrations
    Random.seed!(42)
    
    # Generate synthetic data for Bayesian inference demonstrations
    println("Generating synthetic dataset for Bayesian inference...")
    data = generate_sample(20, β²=0.1)  # 20 observations with low noise
    
    # Demonstrate sequential Bayesian inference in dual parameterizations
    println("Creating Bayesian inference visualizations...")
    println("  - Location-scale (μ, σ²) parameter trajectory")
    plot_Gaussian_inference_μσ²(data, Gaussian.Gaussian1D(0, 1), β²=0.1)
    savefig("~/Downloads/gaussian_inference_μσ².svg")
    
    println("  - Precision-based (τ, ρ) parameter trajectory")
    plot_Gaussian_inference_τρ(data, Gaussian.Gaussian1D(0, 1), β²=0.1)
    savefig("~/Downloads/gaussian_inference_τρ.svg")

    # Create animations showing asymptotic behavior
    println("Creating asymptotic behavior animations...")
    println("  - Dirac delta convergence: N(0, 1/n) → δ(0)")
    anim = plot_dirac_animation()
    gif(anim, "~/Downloads/dirac.gif", fps=10)
    
    println("  - Uniform approximation: N(0, n) → flat distribution")
    anim = plot_uniform_animation()
    gif(anim, "~/Downloads/uniform.gif", fps=10)

    # Demonstrate distribution operations with educational examples
    println("Demonstrating distribution operations...")
    println("  - Multiplication: Bayesian updating example")
    test_multiply(
        Gaussian.NonNormalizedGaussian1D(0, 1, 0.0),      # Neutral prior
        Gaussian.Gaussian1DFromMeanVariance(2, 1.5)       # Offset likelihood
    )
    savefig("~/Downloads/test_gaussian_multiply.pdf")
    
    println("  - Division: Cavity distribution computation")
    test_division(
        Gaussian.NonNormalizedGaussian1D(0, 1, 0.0),      # Starting belief
        Gaussian.Gaussian1DFromMeanVariance(1, 3)         # Factor to remove
    )
    savefig("~/Downloads/test_gaussian_division.pdf")
    
    println("Demonstration complete! Check ~/Downloads/ for all generated files.")
    println("Files created:")
    println("  - gaussian_inference_μσ².svg: Location-scale trajectory")
    println("  - gaussian_inference_τρ.svg: Precision parameter trajectory")
    println("  - dirac.gif: Delta function convergence animation")
    println("  - uniform.gif: Uniform approximation animation")
    println("  - test_gaussian_multiply.pdf: Multiplication demonstration")
    println("  - test_gaussian_division.pdf: Division demonstration")
end

# Export public interface for external use
export generate_sample, plot_Gaussian_inference_μσ², plot_Gaussian_inference_τρ
export plot_dirac_animation, plot_uniform_animation, test_multiply, test_division, main

end
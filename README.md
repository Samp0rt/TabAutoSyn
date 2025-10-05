# TabAutoSyn

A comprehensive tabular data synthesis framework for automated machine learning, featuring advanced genetic algorithms and hyperparameter optimization for generating high-quality synthetic datasets.

## Features

- **Automated Tabular Data Synthesis**: Generate synthetic tabular data using various state-of-the-art models
- **Genetic Algorithm Optimization**: Advanced genetic algorithms for data curation and optimization
- **Hyperparameter Optimization**: Automated hyperparameter tuning using Optuna
- **Multiple Synthesis Models**: Support for task-specific and LLM-based synthesis approaches
- **Comprehensive Metrics**: Extensive evaluation metrics for data quality assessment
- **Privacy-Preserving Synthesis**: Built-in privacy protection mechanisms

## Installation

### From Source

```bash
git clone https://github.com/your-org/tabautosyn.git
cd tabautosyn
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/your-org/tabautosyn.git
cd tabautosyn
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from tabautosyn import TabAutoSyn
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize the synthesizer
synthesizer = TabAutoSyn(
    model="task_specific",
    task="ml",  # or "privacy", "universal"
    verbose=True
)

# Generate synthetic data
synthetic_data = synthesizer.generate(
    data, 
    n_samples=1000
)

print(f"Generated {len(synthetic_data)} synthetic samples")
```

## Project Structure

```
TabAutoSyn/
├── tabautosyn/                 # Main package
│   ├── automl/                 # AutoML components
│   │   └── base.py            # Core TabAutoSyn class
│   ├── config.py              # Metrics configuration
│   ├── custom_metric.py       # Custom evaluation metrics
│   ├── optimization.py        # Hyperparameter optimization
│   └── outliers.py            # Outlier detection and handling
├── curation/                   # Data curation tools
│   ├── gen/                   # Genetic algorithm implementation
│   │   ├── gen.py            # Main genetic algorithm class
│   │   ├── individ.py        # Individual representation
│   │   ├── fitness.py        # Fitness evaluation
│   │   ├── crossover.py      # Crossover operators
│   │   ├── mutation.py       # Mutation operators
│   │   └── selection.py      # Selection operators
│   └── curation_gen.ipynb    # Example notebook
├── datasets/                   # Sample datasets
└── examples/                   # Usage examples
```

## Supported Models

TabAutoSyn supports various synthesis models through the SynthCity framework:

- **CTGAN**: Conditional Tabular GAN
- **TVAE**: Tabular VAE
- **DDPM**: Denoising Diffusion Probabilistic Models
- **DPGAN**: Differentially Private GAN
- **And many more...**

## Evaluation Metrics

The framework provides comprehensive evaluation metrics across multiple categories:

- **Sanity Checks**: Data mismatch, common rows proportion, nearest neighbor distance
- **Statistical Metrics**: Jensen-Shannon distance, Chi-squared test, feature correlation
- **Performance Metrics**: Linear model, MLP, XGBoost performance
- **Detection Metrics**: Detection using various models
- **Privacy Metrics**: Delta-presence, k-anonymization, identifiability score

## Examples

Check out the `examples/` directory for comprehensive usage examples:

- Basic synthesis workflows
- Genetic algorithm optimization
- Hyperparameter tuning
- Privacy-preserving synthesis

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SynthCity](https://github.com/vanderschaarlab/synthcity) for the synthesis framework
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [DEAP](https://github.com/DEAP/deap) for genetic algorithm implementation

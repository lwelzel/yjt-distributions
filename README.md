# Yeo-Johnson Transformed Distributions for PyTorch

This repository provides a PyTorch implementation of the Yeo-Johnson Transformation applied to distributions, allowing for direct sampling from transformed normal distributions. The main use case is to enable sampling from normal distributions transformed using a Yeo-Johnson Transform. Sampling from other base distributions can also be easily implemented by following the provided `YeoJohnsonNormal` class example.

## Features
- **Yeo-Johnson Transform**: Implements the Yeo-Johnson transformation, which can handle both positive and negative inputs, as a PyTorch transform.
- **Transformed Normal Distribution**: Provides a `YeoJohnsonNormal` class for directly sampling from normal distributions transformed with the Yeo-Johnson transform.
- **Extendibility**: Designed to easily extend to other base distributions following the example provided.

## Installation

This package requires PyTorch. You can install the latest version of PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

Once PyTorch is installed, you can clone this repository to your local machine:
```bash
git clone https://github.com/lwelzel/yjt-distributions.git
```

Then, navigate to the repository directory and install the package using `pip`:
```bash
cd yjt-distributions
pip install -e . 
```

## Usage

Here's how to use the YeoJohnsonNormal class to sample from a Yeo-Johnson transformed normal distribution:

```python
import torch
from yjt-distributions import YeoJohnsonNormal

# Parameters for the normal distribution and the transformation
loc = 0.0
scale = 1.0
lbda = 0.5  # Lambda parameter for the Yeo-Johnson transform
tloc = 0.0  # Translation parameter
tscale = 1.0  # Scaling parameter

# Create a YeoJohnsonNormal distribution
yj_normal = YeoJohnsonNormal(loc=torch.tensor([loc]),
                             scale=torch.tensor([scale]),
                             lbda=torch.tensor([lbda]),
                             tloc=torch.tensor([tloc]),
                             tscale=torch.tensor([tscale]))

# Sample from the distribution
samples = yj_normal.sample((1000,))
```

## Extending to Other Distributions

To extend this implementation to other base distributions, you can follow the pattern used in YeoJohnsonNormal. Essentially, you will need to create a new class that inherits from TransformedDistribution and specify your base distribution along with the Yeo-Johnson transform as the transformations to apply.
Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the repository or extend its functionality.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

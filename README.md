# Aperture-aware lens design
## [Project website](https://imaging.cs.cmu.edu/aperture_aware_lens_design/)

This repository includes the code used in the SIGGRAPH 2024 paper "Aperture-aware lens design" by Teh et al.

## Dependencies
The code relies on the following libraries:
* jax
* numpy
* tqdm
* plotly

They can be installed with the command:
    pip install -r requirements.txt

## Examples
In order to run the example code, simply run:

    python -m experiments.run_spot_error

from the source root directory.

## Installation
If interested in installing the code for use with your own code, you can install the "dlt" package with

    pip install .

The package can be then be used by importing dlt
```python
import dlt

# your code here
```

## References
Zemax files are from <https://www.lens-designs.com/>

## Citation
If you use this code, please cite our paper (citation included in the project website).

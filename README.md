# A posteriori error estimates for non-self-consistent Kohn-Sham equations
[![](https://img.shields.io/badge/arxiv-preprint-red)](TODO)

Implementation of the error estimates discussed in the paper:

Michael F. Herbst, Antoine Levitt, Eric Cancès  
*A posteriori error estimation for the non-self-consistent Kohn-Sham equations*  
Preprint on [arxiv](TODO)

The code in this repository has been used to produce all plots
of the above paper. It relies on [DFTK](https://dftk.org)
[version 0.0.6](https://doi.org/10.5281/zenodo.3749552).
For more details see [this blog article](TODO).

## Running the code and reproducing the plots
Running the code requires an installation of
[Julia 1.4.0](https://julialang.org/downloads/#current_stable_release).
With it you can generate the plots of the paper by executing:
```bash
julia --project=@. -e "import Pkg; Pkg.instantiate()"  # Install dependencies
julia run.jl   # Generate data
julia plot.jl  # Generate plots
```

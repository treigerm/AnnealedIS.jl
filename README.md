# AnnealedIS

This is a small package implementing Annealed Importance Sampling. Its key features 
include:
- Integration with [Turing](https://github.com/TuringLang/Turing.jl)
- Specify Markov chain transition kernel using either [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) or 
[AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl)
- Multi-threaded sampling

This implementation here should be seen as a proof of concept. It is not optimized
for performance yet and the API design will likely change in the future.

## Usage example

Integration with Turing:
```julia
using Turing
using AnnealedIS

@model function model(y)
    x ~ Normal(0, 1)
    y ~ Normal(x, 1)
end

y_obs = 2
chains = sample(model(y_obs), AnIS(20), 1000)
```

For more usage examples see the `test/` folder.

## References

Neal, R. M. (1998) "Annealed importance sampling", Technical Report No. 9805 (revised), Dept. of Statistics, University of Toronto, 25 pages
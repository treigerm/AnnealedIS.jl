module AnnealedIS

using Distributions
using AdvancedMH
using LinearAlgebra: I

# Exports
export AnnealedISSampler, ais_sample

# TODO: Clean up types for prior and joint
# TODO: Clean up use of logpdf and logjoint
# TODO: Make it possible to parallelize sampling
# TODO: Better way to specify transition kernels

struct AnnealedISSampler
    prior::Distributions.Distribution
    joint::AdvancedMH.DensityModel
    betas
    transition_kernels
end

function AnnealedISSampler(prior, joint, N::Int)
    betas = collect(range(0, 1, length=N))
    kernel = x -> MvNormal(x) # TODO: Better generic kernel which can handle different types.
    transition_kernels = fill(kernel, N-1)
    return AnnealedISSampler(prior, joint, betas, transition_kernels)
end

"""
One importance sample with associated weight.
"""
struct WeightedSample 
    log_weight
    params
end

"""
Return the importance weight for this sample.
"""
function weight(samples, sampler::AnnealedISSampler)
    N = length(sampler.betas)
    numerator = sum(1:N) do i
        logdensity(sampler, i, samples[i])
    end
    denominator = sum(2:N) do i
        logdensity(sampler, i-1, samples[i])
    end
    denominator += logpdf(sampler.prior, samples[1])
    return numerator - denominator
end

"""
Get the log density of the ith annealed distribution.
"""
function logdensity(sampler::AnnealedISSampler, i, x)
    prior_density = logpdf(sampler.prior, x)
    joint_density = AdvancedMH.logdensity(sampler.joint, x)
    beta = sampler.betas[i]
    return (1-beta) * prior_density + beta * joint_density
end

"""
Sample from the ith transition kernel.
"""
function transition_kernel(rng, sampler::AnnealedISSampler, i, x)
    # The base distribution does not have a transition kernel associated with it.
    @assert i > 1
    
    density(params) = logdensity(sampler, i, params)
    model = DensityModel(density)

    # Do five steps of Metropolis-Hastings
    spl = RWMH(MvNormal(size(x, 1), 0.1125))
    samples = sample(model, spl, 5; progress=false)

    return samples[5].params
end

"""
First MC step.
"""
function single_sample(rng, sampler::AnnealedISSampler)
    # TODO: Do I even need the model?
    N = length(sampler.betas)

    # TODO: Handle multidimensional samples and different types
    prior_sample = rand(rng, sampler.prior) 
    samples = Array{typeof(prior_sample)}(undef, N)

    # Sample from base distribution
    samples[1] = prior_sample
    # Sample from annealed distributions
    for i in 2:N
        samples[i] = transition_kernel(rng, sampler, i, samples[i-1])
    end
    # Get weight of current samples
    w = weight(samples, sampler)
    ws = WeightedSample(w, samples[end])
    return ws
end

function ais_sample(rng, sampler, num_samples)
    first_sample = single_sample(rng, sampler)
    samples = Array{typeof(first_sample)}(undef, num_samples) # TODO: Is this best practice?
    samples[1] = first_sample

    # TODO: Parallelize the loop.
    for i = 2:num_samples
        samples[i] = single_sample(rng, sampler)
    end

    return samples
end

end

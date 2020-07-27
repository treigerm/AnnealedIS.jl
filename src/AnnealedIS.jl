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
    betas = collect(range(0, 1, length=N+1))
    kernel = x -> MvNormal(x) # TODO: Better generic kernel which can handle different types.
    transition_kernels = fill(kernel, N-1)
    return AnnealedISSampler(prior, joint, betas, transition_kernels)
end

"""
One importance sample with associated weight.
"""
# TODO: Type annotations for better performance.
struct WeightedSample 
    log_weight
    params
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
    # The base and target distribution do not have a transition kernel.
    @assert i > 1
    @assert i-1 <= length(sampler.transition_kernels)
    
    density(params) = logdensity(sampler, i, params)
    model = DensityModel(density)

    # Do five steps of Metropolis-Hastings
    spl = RWMH(MvNormal(size(x, 1), 1))
    # TODO: Set x as the initial state.
    samples = sample(model, spl, 5; progress=false, init_params=x)

    return samples[5].params
end

"""
First MC step.
"""
function single_sample(rng, sampler::AnnealedISSampler)
    N = length(sampler.betas)
    num_samples = N-1

    sample = rand(rng, sampler.prior)
    log_weight = logdensity(sampler, 2, sample) - logdensity(sampler, 1, sample)
    for i in 2:num_samples
        sample = transition_kernel(rng, sampler, i, sample)
        log_weight += logdensity(sampler, i+1, sample) - logdensity(sampler, i, sample)
    end

    ws = WeightedSample(log_weight, sample)
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

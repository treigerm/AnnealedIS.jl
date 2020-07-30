module AnnealedIS

using Turing
using Distributions
using AdvancedMH
using LinearAlgebra: I
using Random

# Exports
export AnnealedISSampler, 
    ais_sample, 
    sample_from_prior, 
    make_log_prior_density, 
    make_log_joint_density

# TODO: Make it possible to parallelize sampling; difficulty is thread-safe random number generation.

# TODO: Add types.
struct AnnealedISSampler
    prior_sampling::Function
    prior_density::Function
    joint_density::Function
    betas
    transition_kernels
end

function AnnealedISSampler(prior_sampling, prior_density, joint_density, N::Int)
    betas = collect(range(0, 1, length=N+1))

    prior_sample = prior_sampling(Random.GLOBAL_RNG)
    kernel = get_normal_transition_kernel(prior_sample)
    transition_kernels = fill(kernel, N-1)

    return AnnealedISSampler(
        prior_sampling,
        prior_density, 
        joint_density, 
        betas, 
        transition_kernels
    )
end

function AnnealedISSampler(model::Turing.Model, N::Int)
    return AnnealedISSampler(
        rng -> sample_from_prior(rng, model),
        make_log_prior_density(model),
        make_log_joint_density(model),
        N
    )
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
    prior_density = sampler.prior_density(x)
    joint_density = sampler.joint_density(x)
    beta = sampler.betas[i]

    # NOTE: if beta == 0 and joint_density == -Inf then we get NaN. That's why 
    # we handle the special cases explicitly.
    if beta == 0.0
        return prior_density
    elseif beta == 1.0
        return joint_density
    else
        return (1-beta) * prior_density + beta * joint_density
    end
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
    # TODO: Possible use an iterator approach from AbstractMCMC to avoid saving 
    # unused samples.
    spl = AdvancedMH.RWMH(sampler.transition_kernels[i-1])
    samples = sample(model, spl, 5; progress=false, init_params=x)

    return samples[end].params
end

"""
First MC step.
"""
function single_sample(rng, sampler::AnnealedISSampler)
    N = length(sampler.betas)
    num_samples = N-1

    sample = sampler.prior_sampling(rng)
    # TODO: Assert that logdensity(sampler, 1, sample) is not -Inf because 
    # then our prior sampling or our density is wrong.
    log_weight = logdensity(sampler, 2, sample) - logdensity(sampler, 1, sample)
    if log_weight == -Inf
        # We have an out of distribution sample.
        return WeightedSample(log_weight, sample)
    end

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

function estimate_expectation(samples::Array{WeightedSample}, f)
    weighted_sum = sum(samples) do weighted_sample
        exp(weighted_sample.log_weight) * f(weighted_sample.params)
    end
    sum_weights = sum(samples) do weighted_sample
        exp(weighted_sample.log_weight)
    end
    return weighted_sum / sum_weights
end


##############################
# Transition kernels
##############################

# TODO: How to tune the standard deviation?

# TODO: Is this an appropriate type annotation?
function get_normal_transition_kernel(prior_sample::T) where {T<:Real}
    return Normal(0, 1) 
end

function get_normal_transition_kernel(prior_sample::AbstractArray)
    return MvNormal(size(prior_sample, 1), 1) 
end

function get_normal_transition_kernel(prior_sample::NamedTuple)
    return map(get_normal_transition_kernel, prior_sample)
end

# TODO: Possibly add this to AdvancedMH.jl
AdvancedMH.RWMH(nt::NamedTuple) = MetropolisHastings(map(x -> RandomWalkProposal(x), nt))


##############################
# Turing Interop
##############################

function sample_from_prior(rng, model)
    # TODO: Need to make use of rng.
    # TODO: Need some default VarInfo because this always runs the model once
    #   to get a TypedVarInfo.
    vi = Turing.VarInfo(model) 
    model(vi)

    # Extract a NamedTuple from VarInfo which has variable names as keys and 
    # the sampled values as values.
    # TODO: Check that vi is TypedVarInfo
    vns = Turing.DynamicPPL._getvns(vi, Turing.SampleFromPrior())
    vt = _val_tuple(vi, vns)
    return vt
end

"""
Returns a function which is the prior density.
"""
function make_log_prior_density(model)
    return function logprior_density(named_tuple)
        vi = Turing.VarInfo(model)
        set_namedtuple!(vi, named_tuple)
        model(vi, Turing.SampleFromPrior(), Turing.PriorContext())
        return Turing.getlogp(vi)
    end
end

function make_log_joint_density(model)
    return function logjoint_density(named_tuple)
        vi = Turing.VarInfo(model)
        set_namedtuple!(vi, named_tuple)
        model(vi)
        return Turing.getlogp(vi)
    end
end

##############################
# Functions from https://github.com/TuringLang/Turing.jl/blob/master/src/inference/mh.jl
##############################

# Code from https://github.com/TuringLang/Turing.jl/blob/master/src/inference/mh.jl
function set_namedtuple!(vi::Turing.VarInfo, nt::NamedTuple)
    for (n, vals) in pairs(nt)
        vns = vi.metadata[n].vns
        nvns = length(vns)

        # if there is a single variable only
        if nvns == 1
            # assign the unpacked values
            if length(vals) == 1
                vi[vns[1]] = [vals[1];]
            # otherwise just assign the values
            else
                vi[vns[1]] = [vals;]
            end
        # if there are multiple variables
        elseif vals isa AbstractArray
            nvals = length(vals)
            # if values are provided as an array with a single element
            if nvals == 1
                # iterate over variables and unpacked values
                for (vn, val) in zip(vns, vals[1])
                    vi[vn] = [val;]
                end
            # otherwise number of variables and number of values have to be equal
            elseif nvals == nvns
                # iterate over variables and values
                for (vn, val) in zip(vns, vals)
                    vi[vn] = [val;]
                end
            else
                error("Cannot assign `NamedTuple` to `VarInfo`")
            end
        else
            error("Cannot assign `NamedTuple` to `VarInfo`")
        end
    end
end

# unpack a vector if possible
unvectorize(dists::AbstractVector) = length(dists) == 1 ? first(dists) : dists

# possibly unpack and reshape samples according to the prior distribution 
reconstruct(dist::Distribution, val::AbstractVector) = Turing.DynamicPPL.reconstruct(dist, val)
function reconstruct(
    dist::AbstractVector{<:UnivariateDistribution},
    val::AbstractVector
)
    return val
end
function reconstruct(
    dist::AbstractVector{<:MultivariateDistribution},
    val::AbstractVector
)
    offset = 0
    return map(dist) do d
        n = length(d)
        newoffset = offset + n
        v = val[(offset + 1):newoffset]
        offset = newoffset
        return v
    end
end

@generated function _val_tuple(
    vi::Turing.VarInfo,
    vns::NamedTuple{names}
) where {names}
    isempty(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        :($name = reconstruct(unvectorize(Turing.DynamicPPL.getdist.(Ref(vi), vns.$name)),
                              Turing.DynamicPPL.getval(vi, vns.$name)))
        for name in names]
    return expr
end

end

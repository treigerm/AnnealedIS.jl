module AnnealedIS

using Turing
using Distributions
using AdvancedMH
using LinearAlgebra: I
using StatsFuns: logsumexp
using Random

# Exports
export AnnealedISSampler, 
    RejectionResample,
    SimpleRejection,
    ais_sample, 
    sample_from_prior, 
    make_log_prior_density, 
    make_log_joint_density,
    effective_sample_size

# TODO: Make it possible to parallelize sampling; difficulty is thread-safe random number generation.

abstract type RejectionSampler end

"""
Resample when a sample got rejected.
"""
struct RejectionResample <: RejectionSampler end

struct SimpleRejection <: RejectionSampler end

# TODO: Add types.
struct AnnealedISSampler{S<:RejectionSampler}
    prior_sampling::Function
    prior_density::Function
    joint_density::Function
    betas::Array{Float64}
    transition_kernels
    rejection_sampler::S
end

function AnnealedISSampler(
    prior_sampling::Function, 
    prior_density::Function, 
    joint_density::Function, 
    transition_kernel_fn::Function, 
    N::Int,
    rejection_sampler::T
) where {T<:RejectionSampler}
    betas = collect(range(0, 1, length=N+1))

    prior_sample = prior_sampling(Random.GLOBAL_RNG)
    transition_kernels = [transition_kernel_fn(prior_sample, i) for i in 1:(N-1)]

    return AnnealedISSampler(
        prior_sampling,
        prior_density, 
        joint_density, 
        betas, 
        transition_kernels,
        rejection_sampler
    )
end

function AnnealedISSampler(
    prior_sampling::Function, 
    prior_density::Function, 
    joint_density::Function, 
    transition_kernel_fn::Function, 
    N::Int
)
    return AnnealedISSampler(
        prior_sampling,
        prior_density,
        joint_density,
        transition_kernel_fn,
        N,
        SimpleRejection()
    )
end

function AnnealedISSampler(
    prior_sampling, 
    prior_density, 
    joint_density, 
    N::Int,
    rejection_sampler::T
) where {T<:RejectionSampler}
    return AnnealedISSampler(
        prior_sampling,
        prior_density, 
        joint_density, 
        get_normal_transition_kernel,
        N,
        rejection_sampler
    )
end

function AnnealedISSampler(prior_sampling, prior_density, joint_density, N::Int)
    return AnnealedISSampler(
        prior_sampling,
        prior_density, 
        joint_density, 
        get_normal_transition_kernel,
        N
    )
end

function AnnealedISSampler(
    model::Turing.Model, 
    transition_kernel_fn::Function,
    N::Int
    )
    return AnnealedISSampler(
        rng -> sample_from_prior(rng, model),
        make_log_prior_density(model),
        make_log_joint_density(model),
        transition_kernel_fn,
        N
    )
end

function AnnealedISSampler(model::Turing.Model, N::Int)
    return AnnealedISSampler(
        model,
        get_normal_transition_kernel,
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

function logdensity_ratio(sampler::AnnealedISSampler, i, x)
    prior_density = sampler.prior_density(x)
    joint_density = sampler.joint_density(x)
    beta_i = sampler.betas[i]
    beta_ip1 = sampler.betas[i+1]

    return (beta_ip1 - beta_i) * joint_density - (beta_ip1 - beta_i) * prior_density
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
    # unused samples. NOTE: Requires AbstractMCMC 2
    spl = AdvancedMH.RWMH(sampler.transition_kernels[i-1])
    samples = sample(model, spl, 5; progress=false, init_params=x)

    return samples[end].params
end

function resample(rng, sampler::AnnealedISSampler{SimpleRejection}, num_rejected)
    sample = sampler.prior_sampling(rng)
    log_prior = logdensity(sampler, 1, sample)
    log_transition = logdensity(sampler, 2, sample)
    if (log_prior == -Inf || log_transition == -Inf)
        # Reject sample.
        return WeightedSample(-Inf, sample), true, num_rejected+1
    else
        log_weight = log_transition - log_prior
        return WeightedSample(log_weight, sample), false, num_rejected
    end
end

function resample(rng, sampler::AnnealedISSampler{RejectionResample}, num_rejected)
    sample = sampler.prior_sampling(rng)
    log_prior = logdensity(sampler, 1, sample)
    log_transition = logdensity(sampler, 2, sample)
    if (log_prior == -Inf || log_transition == -Inf)
        # Reject sample.
        return resample(rng, sampler, num_rejected+1)
    else
        log_weight = log_transition - log_prior
        return WeightedSample(log_weight, sample), false, num_rejected
    end
end

function single_sample(
    rng, 
    sampler::AnnealedISSampler; 
    store_intermediate_samples=false
)
    N = length(sampler.betas)
    num_samples = N-1

    # TODO: Make this a named tuple.
    diagnostics = Dict()

    weighted_sample, early_return, num_rejected = resample(rng, sampler, 0)
    diagnostics[:num_rejected] = num_rejected
    log_weight, sample = weighted_sample.log_weight, weighted_sample.params

    if store_intermediate_samples
        diagnostics[:intermediate_samples] = Array{typeof(sample)}(
            undef, 
            Int(num_samples/10)+1
        )
        diagnostics[:intermediate_samples][1] = sample
    end
    if early_return
        return weighted_sample, diagnostics
    end


    for i in 2:num_samples
        sample = transition_kernel(rng, sampler, i, sample)
        log_weight += logdensity_ratio(sampler, i, sample)
        if store_intermediate_samples && (i % 10 == 0)
            ix = Int(i / 10) + 1
            diagnostics[:intermediate_samples][ix] = sample
        end
    end

    ws = WeightedSample(log_weight, sample)
    return ws, diagnostics
end
end
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

function effective_sample_size(samples::Array{WeightedSample}; normalise=true)
    log_weights = map(s -> s.log_weight, samples)
    denominator = logsumexp(2 * log_weights)
    numerator = 1
    if normalise
        numerator = 2 * logsumexp(log_weights)
    end

    return exp(numerator - denominator)
end

##############################
# Transition kernels
##############################

# TODO: How to tune the standard deviation?

# TODO: Is this an appropriate type annotation?
function get_normal_transition_kernel(prior_sample::T, i) where {T<:Real}
    return Normal(0, 1) 
end

function get_normal_transition_kernel(prior_sample::AbstractArray, i)
    return MvNormal(size(prior_sample, 1), 1) 
end

function get_normal_transition_kernel(prior_sample::NamedTuple, i)
    return map(x -> get_normal_transition_kernel(x, i), prior_sample)
end

# TODO: Possibly add this to AdvancedMH.jl
AdvancedMH.RWMH(nt::NamedTuple) = MetropolisHastings(map(x -> RandomWalkProposal(x), nt))

##############################
# Utility functions
##############################

function add_diagnostics!(diagnostics, d, i)
    for k in keys(d)
        diagnostics[k][i] = d[k]
    end
end

function init_diagnostics(first_diagnostics, num_samples)
    diagnostics = Dict()
    for k in keys(first_diagnostics)
        diagnostics[k] = Array{typeof(first_diagnostics[k])}(
        undef, num_samples)
        diagnostics[k][1] = first_diagnostics[k]
    end
    
    return diagnostics
end

##############################
# Turing Interop
##############################

function sample_from_prior(rng, model)
    # TODO: Need to make use of rng.
    vi = Turing.VarInfo(model) 

    # Extract a NamedTuple from VarInfo which has variable names as keys and 
    # the sampled values as values.
    # TODO: Check that vi is TypedVarInfo
    vns = Turing.DynamicPPL._getvns(vi, Turing.SampleFromPrior())
    value_tuple = _val_tuple(vi, vns)
    return value_tuple
end

"""
Returns a function which is the prior density.
"""
function make_log_prior_density(model)
    typed_vi = Turing.VarInfo(model)
    return function logprior_density(named_tuple)
        vi = deepcopy(typed_vi)
        set_namedtuple!(vi, named_tuple)
        model(vi, Turing.SampleFromPrior(), Turing.PriorContext())
        return Turing.getlogp(vi)
    end
end

function make_log_joint_density(model)
    typed_vi = Turing.VarInfo(model)
    return function logjoint_density(named_tuple)
        vi = deepcopy(typed_vi)
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

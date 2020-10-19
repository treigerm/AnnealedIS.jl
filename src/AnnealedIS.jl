module AnnealedIS

using Turing
using Distributions
using AdvancedMH
using AdvancedHMC; const AHMC = AdvancedHMC
using ReverseDiff
using Memoization
using LinearAlgebra: I
using StatsFuns: logsumexp
using Random
using Bijectors

# Exports
export AnnealedISSampler,
    RejectionSampler,
    RejectionResample,
    AnIS,
    AnISHMC,
    AnnealedISSamplerHMC,
    sample,
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

function Base.show(io::IO, anis::AnnealedISSampler)
    s = "AnnealedIS(num_betas=$(length(anis.betas)),rejection=$(anis.rejection_sampler))"
    print(io, s)
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

function AnnealedISSampler(
    model::Turing.Model, 
    transition_kernel_fn::Function,
    N::Int,
    rejection_sampler::T
) where {T<:RejectionSampler}
    return AnnealedISSampler(
        rng -> sample_from_prior(rng, model),
        make_log_prior_density(model),
        make_log_joint_density(model),
        transition_kernel_fn,
        N,
        rejection_sampler
    )
end

function AnnealedISSampler(model::Turing.Model, N::Int)
    return AnnealedISSampler(
        model,
        get_normal_transition_kernel,
        N
    )
end

function AnnealedISSampler(
    model::Turing.Model, 
    N::Int, 
    rejection_sampler::T
) where {T<:RejectionSampler}
    return AnnealedISSampler(
        model,
        get_normal_transition_kernel,
        N,
        rejection_sampler
    )
end

struct AnISModel <: AbstractMCMC.AbstractModel
    prior_sampling::Function
    prior_density::Function
    joint_density::Function
end

function AnISModel(m::Turing.Model)
    return AnISModel(
        rng -> sample_from_prior(rng, m),
        make_log_prior_density(m),
        make_log_joint_density(m)
    )
end

function AnISModel(m::Turing.Model, vi)
    return AnISModel(
        rng -> sample_from_prior(rng, m),
        make_log_prior_density(m, vi),
        make_log_joint_density(m, vi)
    )
end

struct AnIS{S<:RejectionSampler} <: Turing.InferenceAlgorithm
    betas::Array{Float64}
    transition_kernel_fn::Function
    rejection_sampler::S
end

function AnIS(
    transition_kernel_fn, 
    N, 
    rejection_sampler::S
) where {S<:RejectionSampler}
    betas = collect(range(0, 1, length=N+1))
    return AnIS(
        betas,
        transition_kernel_fn,
        rejection_sampler
    )
end

function AnIS(N)
    return AnIS(
        get_normal_transition_kernel, 
        N,
        SimpleRejection()
    )
end

function Base.show(io::IO, anis::AnIS)
    s = "AnnealedIS(num_betas=$(length(anis.betas)),rejection=$(anis.rejection_sampler))"
    print(io, s)
end


function AnnealedISSampler(model::AnISModel, alg::AnIS)
    N = length(alg.betas)-1
    prior_sample = model.prior_sampling(Random.GLOBAL_RNG)
    transition_kernels = [
        alg.transition_kernel_fn(prior_sample, i) for i in 1:(N-1)
    ]
    return AnnealedISSampler(
        model.prior_sampling,
        model.prior_density,
        model.joint_density,
        alg.betas,
        transition_kernels,
        alg.rejection_sampler
    )
end

function AnnealedISSampler(model::Turing.Model, alg::AnIS)
    return AnnealedISSampler(AnISModel(model), alg)
end

function AnnealedISSampler(model::Turing.Model, alg::AnIS, vi)
    return AnnealedISSampler(AnISModel(model, vi), alg)
end

# HMC initialisation.

struct AnISHMC{AD,S<:RejectionSampler,P<:AHMC.AbstractProposal} <: Turing.Hamiltonian{AD}
    betas::Array{Float64}
    proposal::P
    num_samples::Int # Number of samples for single transition kernel
    rejection_sampler::S
end

# Set ForwardDiffAD as default AD
AnISHMC(args...; kwargs...) = AnISHMC{Turing.Core.ForwardDiffAD{1}}(args...; kwargs...)
function AnISHMC{AD}(
    betas, 
    proposal::P, 
    num_samples, 
    rejection_sampler::S
) where {AD, S<:RejectionSampler, P<:AHMC.AbstractProposal}
    return AnISHMC{AD,S,P}(betas, proposal, num_samples, rejection_sampler)
end

Turing.DynamicPPL.getspace(::AnISHMC) = ()
# NOTE: ReverseDiff will give wrong results because support for using contexts is broken.
#Turing.Core.getADbackend(alg::AnISHMC) = Turing.Core.ReverseDiffAD{true}()
#Turing.Core.getADbackend(alg::AnISHMC) = Turing.ForwardDiffAD{1}()
Turing.Core.getADbackend(alg::AnISHMC{AD,S,P}) where {AD,S,P} = AD()
Turing.Core.getchunksize(::Type{<:AnISHMC{AD,S,P}}) where {AD,S,P} = Turing.Core.getchunksize(AD)

struct AnnealedISSamplerHMC{AD,S<:RejectionSampler,P<:AHMC.AbstractProposal}
    prior_sampling::Function
    prior_density::Function
    joint_density::Function
    prior_grad::Function
    joint_grad::Function
    hamiltonians::Array{AHMC.Hamiltonian,1}
    alg::AnISHMC{AD,S,P}
end

function AnnealedISSamplerHMC(model::Turing.Model, alg::AnISHMC, spl)
    prior_sampling = gen_prior_sample(model)
    logprior = gen_logprior(spl.state.vi, model, spl)
    logjoint = gen_logjoint(spl.state.vi, model, spl)

    logprior_grad = gen_logprior_grad(spl.state.vi, model, spl)
    logjoint_grad = gen_logjoint_grad(spl.state.vi, model, spl)

    hamiltonians = make_hamiltonians(
        alg.betas,
        logprior, 
        logjoint, 
        logprior_grad, 
        logjoint_grad,
        length(spl.state.vi[spl])
    )
    return AnnealedISSamplerHMC(
        prior_sampling, 
        logprior,
        logjoint,
        logprior_grad,
        logjoint_grad,
        hamiltonians,
        alg
    )
end

function make_hamiltonians(
    betas,
    logprior,
    logjoint,
    logprior_grad,
    logjoint_grad,
    dim
)
    hamiltonians = Array{AHMC.Hamiltonian,1}(undef,length(betas)-1)
    # We don't need a hamiltonian for the first beta (=0).
    for (i, b) in enumerate(betas[2:end])
        density(x) = (1 - b) * logprior(x) + b * logjoint(x)
        density_grad(x) = (1 - b) .* logprior_grad(x) .+ b .* logjoint_grad(x)

        metric = AHMC.UnitEuclideanMetric(dim)
        hamiltonians[i] = AHMC.Hamiltonian(metric, density, density_grad)
    end

    return hamiltonians
end

"""
One importance sample with associated weight.
"""
# TODO: Type annotations for better performance.
# TODO: Possibly rename params into θ and add lp field to be consistent with Turing.
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

function logdensity(sampler::AnnealedISSamplerHMC, i, x)
    prior_density = sampler.prior_density(x)
    joint_density = sampler.joint_density(x)
    beta = sampler.alg.betas[i]

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

function logdensity_ratio(sampler::AnnealedISSamplerHMC, i, x)
    prior_density = sampler.prior_density(x)
    joint_density = sampler.joint_density(x)
    beta_i = sampler.alg.betas[i]
    beta_ip1 = sampler.alg.betas[i+1]

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
    num_steps = 20
    #num_steps = i > 2 ? 20 : 1000
    samples = sample(model, spl, num_steps; progress=false, init_params=x)

    return samples[end].params
end

function transition_kernel(rng, sampler::AnnealedISSamplerHMC, i, x)

    hamiltonian = sampler.hamiltonians[i-1]
    proposal = sampler.alg.proposal

    samples, _ = sample(hamiltonian, proposal, x, sampler.alg.num_samples; verbose=false)
    return samples[end]
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

function resample(
    rng, 
    sampler::AnnealedISSamplerHMC{AD,SimpleRejection,P}, 
    num_rejected
) where {AD,P<:AHMC.AbstractProposal}
    sample = sampler.prior_sampling()
    log_prior = logdensity(sampler, 1, sample)
    log_transition = logdensity(sampler, 2, sample)
    if (log_prior == -Inf || log_transition == -Inf)
        # Reject sample.
        @show sample
        @show log_prior
        @show log_transition
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
        # FIXME: Number of annealing distributions has to be multiple of 10.
        diagnostics[:intermediate_samples] = Array{typeof(sample)}(
            undef, 
            Int(num_samples/10)+9
        )
        diagnostics[:intermediate_log_weights] = Array{typeof(log_weight)}(
            undef, 
            Int(num_samples/10)+9
        )
        diagnostics[:intermediate_samples][1] = sample
        diagnostics[:intermediate_log_weights][1] = log_weight
    end
    if early_return
        return weighted_sample, diagnostics
    end


    for i in 2:num_samples
        sample = transition_kernel(rng, sampler, i, sample)
        log_weight += logdensity_ratio(sampler, i, sample)
        if store_intermediate_samples && (i % 10 == 0 || i < 10)
            ix = i < 10 ? i : Int(i / 10) + 9
            diagnostics[:intermediate_samples][ix] = sample
            diagnostics[:intermediate_log_weights][ix] = log_weight
        end
    end

    ws = WeightedSample(log_weight, sample)
    return ws, diagnostics
end

function single_sample(
    rng, 
    sampler::AnnealedISSamplerHMC; 
    store_intermediate_samples=false
)
    N = length(sampler.alg.betas)
    num_samples = N-1

    # TODO: Make this a named tuple.
    diagnostics = Dict()

    weighted_sample, early_return, num_rejected = resample(rng, sampler, 0)
    diagnostics[:num_rejected] = num_rejected
    log_weight, sample = weighted_sample.log_weight, weighted_sample.params

    if store_intermediate_samples
        # FIXME: Number of annealing distributions has to be multiple of 10.
        diagnostics[:intermediate_samples] = Array{typeof(sample)}(
            undef, 
            Int(num_samples/10)+9
        )
        diagnostics[:intermediate_log_weights] = Array{typeof(log_weight)}(
            undef, 
            Int(num_samples/10)+9
        )
        diagnostics[:intermediate_samples][1] = sample
        diagnostics[:intermediate_log_weights][1] = log_weight
    end
    if early_return
        return weighted_sample, diagnostics
    end


    for i in 2:num_samples
        sample = transition_kernel(rng, sampler, i, sample)
        log_weight += logdensity_ratio(sampler, i, sample)
        if store_intermediate_samples && (i % 10 == 0 || i < 10)
            ix = i < 10 ? i : Int(i / 10) + 9
            diagnostics[:intermediate_samples][ix] = sample
            diagnostics[:intermediate_log_weights][ix] = log_weight
        end
    end

    ws = WeightedSample(log_weight, sample)
    return ws, diagnostics
end

function ais_sample(rng, sampler, num_samples; store_intermediate_samples=false)
    first_sample, first_diagnostics = single_sample(
        rng, 
        sampler;
        store_intermediate_samples=store_intermediate_samples
    )
    samples = Array{typeof(first_sample)}(undef, num_samples)
    samples[1] = first_sample
    
    diagnostics = init_diagnostics(first_diagnostics, num_samples)

    # TODO: Parallelize the loop.
    spls = [deepcopy(sampler) for _ in 1:Threads.nthreads()]
    Threads.@threads for i = 2:num_samples
    #println("Not parallel")
    #for i = 2:num_samples
        tid = Threads.threadid()
        samples[i], d = single_sample(
            rng, 
            spls[tid];
            store_intermediate_samples=store_intermediate_samples
        )
        add_diagnostics!(diagnostics, d, i)
    end

    diagnostics[:ess] = effective_sample_size(samples)

    return samples, diagnostics
end

function ais_sample(
    rng, 
    model::AnISModel, 
    sampler::AnIS, 
    num_samples; 
    kwargs...
)
    ais = AnnealedISSampler(model, sampler)
    return ais_sample(rng, ais, num_samples; kwargs...)
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

include("turing_interop.jl")

####
#### Compiler interface, i.e. tilde operators.
####
#### Taken from https://github.com/TuringLang/Turing.jl/blob/master/src/inference/hmc.jl
function DynamicPPL.assume(
    rng,
    spl::Sampler{<:AnISHMC},
    dist::Distribution,
    vn::VarName,
    vi,
)
    updategid!(vi, vn, spl)
    r = vi[vn]
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    # r
    return r, Bijectors.logpdf_with_trans(dist, r, istrans(vi, vn))
end

function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:AnISHMC},
    dist::MultivariateDistribution,
    vns::AbstractArray{<:VarName},
    var::AbstractMatrix,
    vi,
)
    @assert length(dist) == size(var, 1)
    updategid!.(Ref(vi), vns, Ref(spl))
    r = vi[vns]
    var .= r
    return var, sum(Bijectors.logpdf_with_trans(dist, r, istrans(vi, vns[1])))
end
function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:AnISHMC},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi,
)
    updategid!.(Ref(vi), vns, Ref(spl))
    r = reshape(vi[vec(vns)], size(var))
    var .= r
    return var, sum(Bijectors.logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
end

function DynamicPPL.observe(
    spl::Sampler{<:AnISHMC},
    d::Distribution,
    value,
    vi,
)
    return DynamicPPL.observe(SampleFromPrior(), d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Sampler{<:AnISHMC},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(SampleFromPrior(), ds, value, vi)
end

end
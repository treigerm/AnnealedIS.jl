module AnnealedIS

using Turing
using Distributions
using AdvancedMH
using LinearAlgebra: I
using StatsFuns: logsumexp
using Random

# Exports
export AnnealedISSampler, 
    RejectionSampler,
    RejectionResample,
    AnIS,
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

function AnnealedISSampler(model::AnISModel, alg::AnIS)
    return AnnealedISSampler(
        model.prior_sampling,
        model.prior_density,
        model.joint_density,
        alg.transition_kernel_fn,
        length(alg.betas),
        alg.rejection_sampler
    )
end

function AnnealedISSampler(model::Turing.Model, alg::AnIS)
    return AnnealedISSampler(AnISModel(model), alg)
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
    for i = 2:num_samples
        samples[i], d = single_sample(
            rng, 
            sampler; 
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

##############################
# Turing Interop
##############################

function sample_from_prior(rng, model)
    # TODO: Need to make use of rng.
    vi = Turing.VarInfo(model) 

    # Extract a NamedTuple from VarInfo which has variable names as keys and 
    # the sampled values as values.
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

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::T,
    alg::AnIS,
    num_samples::Int;
    kwargs...
) where {T <: Union{Turing.Model,AnISModel}}
    anis = AnnealedISSampler(model, alg)
    samples, diagnostics = ais_sample(rng, anis, num_samples; kwargs...)
    samples = make_turing_samples(samples, model)
    return to_mcmcchains(samples, diagnostics)
end

function make_turing_samples(samples, model::Turing.Model)
    vi = Turing.VarInfo(model)
    return map(samples) do sample 
        set_namedtuple!(vi, sample.params)
        nt = DynamicPPL.tonamedtuple(vi)
        (θ = nt, log_weight = sample.log_weight)
    end
end

function to_mcmcchains(samples, diagnostics)
    # Part of this function is adapted from the generic Turing bundle_samples at
    # https://github.com/TuringLang/Turing.jl/blob/60563a64c51f2ea465f85d819344be00d0186d1b/src/inference/Inference.jl

    # Convert transitions to array format.
    # Also retrieve the variable names.
    nms, vals = _params_to_array(samples)

    # Get the values of the extra parameters in each transition.
    extra_params, extra_values = get_transition_extras(samples)

    # Extract names & construct param array.
    nms = [nms; extra_params]
    parray = hcat(vals, extra_values)

    # Calculate logevidence.
    le = logsumexp(map(s -> s.log_weight, samples)) - log(length(samples))

    info = Dict{Symbol,Any}(
        :ess => diagnostics[:ess],
    )
    if haskey(diagnostics, :intermediate_samples)
        info[:intermediate_samples] = diagnostics[:intermediate_samples]
    end
    info = (;info...)

    # Conretize the array before giving it to MCMCChains.
    parray = MCMCChains.concretize(parray)

    return MCMCChains.Chains(
        parray,
        string.(nms),
        (internals = ["log_weight"],);
        evidence=le,
        info=info
    )
end

##############################
# Functions adapted from https://github.com/TuringLang/Turing.jl/blob/60563a64c51f2ea465f85d819344be00d0186d1b/src/inference/Inference.jl
##############################

"""
    getparams(t)
Return a named tuple of parameters.
"""
getparams(t) = t.θ

function _params_to_array(ts::Vector)
    names = Vector{String}()
    # Extract the parameter names and values from each transition.
    dicts = map(ts) do t
        nms, vs = flatten_namedtuple(getparams(t))
        for nm in nms
            if !(nm in names)
                push!(names, nm)
            end
        end
        # Convert the names and values to a single dictionary.
        return Dict(nms[j] => vs[j] for j in 1:length(vs))
    end
    # names = collect(names_set)
    vals = [get(dicts[i], key, missing) for i in eachindex(dicts), 
        (j, key) in enumerate(names)]

    return names, vals
end

function flatten_namedtuple(nt::NamedTuple)
    names_vals = mapreduce(vcat, keys(nt)) do k
        v = nt[k]
        if length(v) == 1
            return [(String(k), v)]
        else
            return mapreduce(vcat, zip(v[1], v[2])) do (vnval, vn)
                return collect(FlattenIterator(vn, vnval))
            end
        end
    end
    return [vn[1] for vn in names_vals], [vn[2] for vn in names_vals]
end

additional_parameters(::Type{<:NamedTuple}) = [:log_weight]

function get_transition_extras(ts::AbstractVector)
    # Get the extra field names from the sampler state type.
    # This handles things like :lp or :weight.
    extra_params = additional_parameters(eltype(ts))

    # Get the values of the extra parameters.
    local extra_names
    all_vals = []

    # Iterate through each transition.
    for t in ts
        extra_names = Symbol[]
        vals = []

        # Iterate through each of the additional field names
        # in the struct.
        for p in extra_params
            # Check whether the field contains a NamedTuple,
            # in which case we need to iterate through each
            # key/value pair.
            prop = getproperty(t, p)
            if prop isa NamedTuple
                for (k, v) in pairs(prop)
                    push!(extra_names, Symbol(k))
                    push!(vals, v)
                end
            else
                push!(extra_names, Symbol(p))
                push!(vals, prop)
            end
        end
        push!(all_vals, vals)
    end

    # Convert the vector-of-vectors to a matrix.
    valmat = [all_vals[i][j] for i in 1:length(ts), j in 1:length(all_vals[1])]

    return extra_names, valmat
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

##############################
# Functions from https://github.com/TuringLang/Turing.jl/blob/515c4d5316d52619388557af8f850740ae310122/src/utilities/helper.jl
##############################

struct FlattenIterator{Tname, Tvalue}
    name::Tname
    value::Tvalue
end

Base.length(iter::FlattenIterator) = _length(iter.value)
_length(a::AbstractArray) = sum(_length, a)
_length(a::AbstractArray{<:Number}) = length(a)
_length(::Number) = 1

Base.eltype(iter::FlattenIterator{String}) = Tuple{String, _eltype(typeof(iter.value))}
_eltype(::Type{TA}) where {TA <: AbstractArray} = _eltype(eltype(TA))
_eltype(::Type{T}) where {T <: Number} = T

@inline function Base.iterate(iter::FlattenIterator{String, <:Number}, i = 1)
    i === 1 && return (iter.name, iter.value), 2
    return nothing
end
@inline function Base.iterate(
    iter::FlattenIterator{String, <:AbstractArray{<:Number}}, 
    ind = (1,),
)
    i = ind[1]
    i > length(iter.value) && return nothing
    name = getname(iter, i)
    return (name, iter.value[i]), (i+1,)
end
@inline function Base.iterate(
    iter::FlattenIterator{String, T},
    ind = startind(T),
) where {T <: AbstractArray}
    i = ind[1]
    i > length(iter.value) && return nothing
    name = getname(iter, i)
    local out
    while i <= length(iter.value)
        v = iter.value[i]
        out = iterate(FlattenIterator(name, v), Base.tail(ind))
        out !== nothing && break
        i += 1
    end
    if out === nothing
        return nothing
    else
        return out[1], (i, out[2]...)
    end
end

@inline startind(::Type{<:AbstractArray{T}}) where {T} = (1, startind(T)...)
@inline startind(::Type{<:Number}) = ()
@inline startind(::Type{<:Any}) = throw("Type not supported.")
@inline function getname(iter::FlattenIterator, i::Int)
    name = string(ind2sub(size(iter.value), i))
    name = replace(name, "(" => "[");
    name = replace(name, ",)" => "]");
    name = replace(name, ")" => "]");
    name = iter.name * name
    return name
end
@inline ind2sub(v, i) = Tuple(CartesianIndices(v)[i])

end
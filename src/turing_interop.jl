##############################
# Turing Interop
##############################

using Turing.DynamicPPL

function Turing.DynamicPPL.Sampler(
    alg::AnISHMC,
    model::Turing.Model,
    s::Turing.DynamicPPL.Selector=Turing.DynamicPPL.Selector()
)
    info = Dict{Symbol,Any}()
    initial_state = Turing.Inference.SamplerState(Turing.VarInfo(model))

    return Turing.DynamicPPL.Sampler(alg, info, s, initial_state)
end

# From AIS code in Turing PR.
"""
    gen_logjoint(v, model, spl)
Return the log joint density function corresponding to model.
"""
function gen_logjoint(v, model, spl)
    function logjoint(z)::Float64
        z_old, lj_old = v[spl], getlogp(v) 
        v[spl] = z
        model(v, spl)
        lj = getlogp(v)
        v[spl] = z_old
        setlogp!(v, lj_old)
        return lj
    end
    return logjoint
end

function gen_logjoint_grad(vi, model, spl::Turing.DynamicPPL.Sampler)
    function logjoint_grad(x)
        return Turing.Core.gradient_logp(
            x, vi, model, spl
        )
    end
    return logjoint_grad
end

"""
    gen_logprior(v, model, spl)
Return the log prior density function corresponding to model.
"""
function gen_logprior(v, model, spl)
    function logprior(z)::Float64
        z_old, lj_old = v[spl], getlogp(v)
        v[spl] = z
        model(v, Turing.SampleFromPrior(), Turing.DynamicPPL.PriorContext())
        lj = getlogp(v)
        v[spl] = z_old
        setlogp!(v, lj_old)
        return lj
    end
    return logprior
end

function gen_logprior_grad(v, model, spl)
    function logprior_grad(x)
        return Turing.Core.gradient_logp(
            x, v, model, spl, Turing.DynamicPPL.PriorContext()
        )
    end
    return logprior_grad
end

function sample_from_prior(rng, model)
    # TODO: Need to make use of rng.
    vi = Turing.VarInfo(model) 

    # Extract a NamedTuple from VarInfo which has variable names as keys and 
    # the sampled values as values.
    vns = Turing.DynamicPPL._getvns(vi, Turing.SampleFromPrior())
    value_tuple = _val_tuple(vi, vns)
    return value_tuple
end

function gen_prior_sample(model)
    spl = Turing.SampleFromPrior()
    function prior_sampling()
        vi = Turing.VarInfo(model)
        z = vi[spl]
        return z
    end

    return prior_sampling
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

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::T,
    alg::AnISHMC,
    num_samples::Int;
    kwargs...
) where {T <: Union{Turing.Model,AnISModel}}
    anis = AnnealedISSamplerHMC(model, alg)
    samples, diagnostics = ais_sample(rng, anis, num_samples; kwargs...)
    samples = make_turing_samples(samples, model, Turing.DynamicPPL.Sampler(alg, model))
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

function make_turing_samples(samples, model::Turing.Model, spl::Turing.DynamicPPL.Sampler)
    vi = spl.state.vi
    return map(samples) do sample 
        vi[spl] = sample.params
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
        info[:intermediate_log_weights] = diagnostics[:intermediate_log_weights]
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
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
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
    return var, sum(logpdf_with_trans(dist, r, istrans(vi, vns[1])))
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
    return var, sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
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
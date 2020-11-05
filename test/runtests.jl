using Turing
using Distributions
using AdvancedMH
using AdvancedHMC
using LinearAlgebra: I, dot
using Random
using Test

using AnnealedIS

function geomspace(start, stop, length)
    logstart = log10(start)
    logstop = log10(stop)
    points = 10 .^ range(logstart, logstop; length=length)
    points[1] = start
    points[end] = stop
    return points
end

@testset "AnnealedIS.jl" begin
    rng = MersenneTwister(42)
    Random.seed!(42)

    @testset "AnnealedISSampler" begin
        D = 5
        N = 2

        prior_sampling(rng) = rand(rng, MvNormal(D, 1.0))
        prior_density(params) = logpdf(MvNormal(D, 1.0), params)
        joint_density(params) = logpdf(MvNormal(D, 0.5), params)

        ais = AnnealedISSampler(
            prior_sampling, prior_density, joint_density, N)

        @test ais.betas == [0.0, 0.5, 1.0]
        x = ais.prior_sampling(rng)
        y = rand(rng, ais.transition_kernels[1])
        @test typeof(x) == typeof(y)

        samples = [ais.prior_sampling(rng) for _ in 1:N]

        @test AnnealedIS.logdensity(ais, 1, samples[1]) == prior_density(samples[1])
        @test AnnealedIS.logdensity(ais, 3, samples[1]) == joint_density(samples[1])
        @test AnnealedIS.logdensity(ais, 2, samples[1]) == (0.5*prior_density(samples[1]) + 0.5*joint_density(samples[1]))

        for i in 1:N
            dratio = AnnealedIS.logdensity(ais, i+1, samples[1]) - AnnealedIS.logdensity(ais, i, samples[1])
            @test AnnealedIS.logdensity_ratio(ais, i, samples[1]) == dratio
        end
    end

    @testset "Convergence test" begin
        D = 1
        N = 100
        num_samples = 1000

        y_obs = [3.0]

        prior_sampling(rng) = rand(rng, MvNormal(D, 1.0))
        prior_density(params) = logpdf(MvNormal(D, 1.0), params)
        joint_density(params) = logpdf(MvNormal(params, I), y_obs) + prior_density(params)

        ais = AnnealedISSampler(
            prior_sampling, prior_density, joint_density, N)

        samples, diagnostics = ais_sample(rng, ais, num_samples)

        posterior_mean = AnnealedIS.estimate_expectation(samples, x -> x)
        @test isapprox(posterior_mean, [1.5]; atol=3e-2)

        ess = effective_sample_size(samples)
        # TODO: What is a reasonable ESS to expect?
        @test ess > 900
        @test ess == diagnostics[:ess]
    end

    @testset "Logdensity ratio" begin
        D = 1
        N = 100
        num_samples = 1000

        y_obs = [3.0]

        prior_sampling(rng) = rand(rng, MvNormal(D, 1.0))
        prior_density(params) = logpdf(MvNormal(D, 1.0), params)
        joint_density(params) = logpdf(MvNormal(params, I), y_obs) + prior_density(params)

        betas = (geomspace(1, 1001, N+1) .- 1) ./ 1000
        anis_alg = 
        ais = AnnealedISSampler(
            AnnealedIS.AnISModel(prior_sampling, prior_density, joint_density), 
            AnIS(betas, (x, i) -> Normal(0, 1), SimpleRejection())
        )
        
        sample = [1.0]
        
        for i in 1:(length(ais.betas)-1)
            @test isapprox(
                AnnealedIS.logdensity_ratio(ais, i, sample),
                (AnnealedIS.logdensity(ais, i+1, sample) - AnnealedIS.logdensity(ais, i, sample)); 
                atol=1e-12 # TODO: What is the numerical error we would expect?
            )
        end
    end

    @testset "Sample from Turing" begin
        @model function test_model(y)
            x ~ Normal(0, 1)
            y ~ Normal(y, 1)
        end

        y_obs = 2
        tm = test_model(y_obs)

        named_tuple = sample_from_prior(rng, tm)
        named_tuple2 = sample_from_prior(rng, tm)
        @test typeof(named_tuple[:x]) == Float64
        @test named_tuple != named_tuple2

        # Test model with multiple latents and multivariate latent.
        D = 5
        @model function test_model2(y)
            a ~ Normal(0, 1)
            x ~ MvNormal(D, 1.0)
            y ~ MvNormal(x, I)
        end

        y_obs = ones(D)
        tm2 = test_model2(y_obs)

        named_tuple = sample_from_prior(rng, tm2)
        @test typeof(named_tuple[:a]) == Float64
        @test typeof(named_tuple[:x]) == typeof(rand(MvNormal(D, 1.0)))
    end

    @testset "Prior density from Turing" begin
        D = 5

        @model function test_model(y)
            a ~ MvNormal(D, 1)
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
        end

        y_obs = 2
        tm = test_model(y_obs)
        logprior_density = make_log_prior_density(tm)

        xval = 1
        aval = ones(D)
        nt = (x = xval, a = aval)
        prior_hand(nt) = logpdf(Normal(0, 1), nt[:x]) + logpdf(MvNormal(D, 1), nt[:a])
        @test logprior_density(nt) == prior_hand(nt)

        # Test that we can evaluate samples from prior
        nt = sample_from_prior(rng, tm)
        @test logprior_density(nt) == prior_hand(nt)
    end

    @testset "Joint density from Turing" begin
        @model function test_model(y)
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
        end

        y_obs = 2
        tm = test_model(y_obs)
        logjoint_density = make_log_joint_density(tm)

        xval = 1
        nt = (x = xval,)
        @test logjoint_density(nt) == logpdf(Normal(0, 1), xval) + logpdf(Normal(xval, 1), y_obs)
    end

    @testset "Turing AnnealedISSampler" begin
        N = 3
        num_samples = 10

        @model function test_model(y)
            x ~ Normal(0, 1)
            z ~ MvNormal(zeros(2))
            y ~ Normal(x, 1)
        end

        y_obs = 2
        tm = test_model(y_obs)

        ais = AnnealedISSampler(tm, N)
        samples, diagnostics = ais_sample(rng, ais, num_samples)

        chains = sample(rng, tm, AnIS(N), num_samples)

        @test typeof(chains) <: MCMCChains.Chains
        @test typeof(chains.logevidence) <: Real
        @test haskey(chains.info, :ess)
        @test haskey(get(chains, :log_weight), :log_weight)
        @test haskey(get(chains, :x), :x)

        # Test intermediate samples
        chains = sample(
            rng, tm, AnIS(20), num_samples; store_intermediate_samples=true
        )
        @test haskey(chains.info, :intermediate_samples)

        inter_samples = hcat(chains.info[:intermediate_samples]...)
        inter_samples = permutedims(inter_samples, [2, 1])
        @test size(inter_samples) == (num_samples, 11)

        @test haskey(chains.info, :intermediate_log_weights)
        inter_weights = hcat(chains.info[:intermediate_log_weights]...)
        inter_weights = permutedims(inter_weights, [2, 1])
        @test size(inter_weights) == (num_samples, 11)
    end

    @testset "Transition kernels" begin
        prior_sample = (a = 0.0, b = [1; 2])
        expected_transition_kernel = (a = Normal(0, 1), b = MvNormal(2, 1.0))
        transition_kernel = AnnealedIS.get_normal_transition_kernel(prior_sample, 1)
        @test expected_transition_kernel == transition_kernel
    end

    @testset "Intermediate Samples" begin
        N = 20 
        num_samples = 10

        @model function test_model(y)
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
        end

        y_obs = 2
        tm = test_model(y_obs)

        ais = AnnealedISSampler(tm, N) 
        samples, diagnostics = ais_sample(
            rng, ais, num_samples; store_intermediate_samples=true)

        inter_samples = hcat(diagnostics[:intermediate_samples]...)
        inter_samples = permutedims(inter_samples, [2, 1])
        @test size(inter_samples) == (num_samples, 11)
        @test map(x->x.params, samples) == inter_samples[:,end]

        @test haskey(diagnostics, :intermediate_log_weights)
    end

    @testset "Resampling Tests" begin
        D = 1
        N = 10
        num_samples = 100

        y_obs = [3.0]

        # Roughly half the samples should get rejected.
        prior_sampling(rng) = rand(rng, MvNormal(D, 1.0))
        prior_density(params) = logpdf(MvNormal(D, 1.0), params)
        joint_density(params) = logpdf(MvNormal(params, I), y_obs) + prior_density(params) + log(max(params[1],0))

        ais_simple_rejection = AnnealedISSampler(
            prior_sampling, 
            prior_density, 
            joint_density, 
            N
        )
        samples_no_resampling, no_resampling_diag = ais_sample(
            rng, 
            ais_simple_rejection, 
            num_samples
        )

        ais_resampling = AnnealedISSampler(
            prior_sampling, 
            prior_density, 
            joint_density, 
            N,
            RejectionResample()
        )
        samples_resampling, resampling_diag = ais_sample(rng, ais_resampling, num_samples)

        non_zero_weights(samples) = sum(samples) do weighted_sample
            Int(weighted_sample.log_weight != -Inf)
        end

        @test non_zero_weights(samples_resampling) == num_samples
        @test sum(resampling_diag[:num_rejected]) > 0

        num_accepted_samples = non_zero_weights(samples_no_resampling)
        @test num_accepted_samples < num_samples
        @test (num_samples - sum(no_resampling_diag[:num_rejected])) == num_accepted_samples
        @test maximum(no_resampling_diag[:num_rejected]) == 1
    end

    @testset "Out of distribution catch" begin
        @model function model()
            x ~ LogNormal(1, 1)
        end

        prior_density = make_log_prior_density(model())
        joint_density = make_log_joint_density(model())

        @test prior_density((x = -1,)) == -Inf
        @test joint_density((x = -1,)) == -Inf
    end

    @testset "AnISHMC" begin
        betas = [0.0, 0.5, 1.0]
        proposal = AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(0.05), 10)
        anis = AnISHMC(
            betas,
            proposal,
            10,
            SimpleRejection()
        )
        @test anis.betas[1] == 0.0
    end

    @testset "AnnealedISSamplerHMC" begin
        N = 100
        num_samples = 10

        @model function test_model(y)
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
        end
        
        yval = 3
        tm = test_model(yval)

        #betas = collect(range(0, 1, length=100))
        betas = (geomspace(1, 1001, 101) .- 1) ./ 1000
        proposal = AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(0.05), 10)
        anis = AnISHMC(
            betas,
            proposal,
            5,
            SimpleRejection()
        )
        spl = Turing.DynamicPPL.Sampler(anis, tm)
        anis_sampler = AnnealedISSamplerHMC(tm, anis, spl)

        # Check that hamiltonians are correct. The first hamiltonian should 
        # have the target density specified by the first non-zero beta.
        prior_sample = anis_sampler.prior_sampling()
        for i in 2:(length(betas)-1)
            h = anis_sampler.hamiltonians[i-1]
            d = AnnealedIS.logdensity(anis_sampler, i, prior_sample)
            @test h.ℓπ(prior_sample) == d

            dr = AnnealedIS.logdensity_ratio(anis_sampler, i, prior_sample)
            hp1 = anis_sampler.hamiltonians[i]
            hdr = hp1.ℓπ(prior_sample) - h.ℓπ(prior_sample)
            @test isapprox(dr, hdr, atol=1e-12)
        end

        chain = sample(
            Random.GLOBAL_RNG, 
            test_model(yval), 
            anis, 
            num_samples;
            store_intermediate_samples=true
        )

        acceptance_rates = chain.info[:intermediate_acceptance_rate]
        @test haskey(chain.info, :intermediate_acceptance_rate)
        @test haskey(chain.info, :acceptance_rate)

        #samples = Array(chain[:x])[:,1]
        #log_weights = Array(chain[:log_weight])[:,1]

        #posterior_mean = dot(samples, exp.(log_weights)) / (num_samples * exp(chain.logevidence))
        #@test isapprox(posterior_mean, 1.5; atol=3e-2)
    end

    # @testset "Constrained Variables" begin
    #     @model function test_model(y)
    #         x ~ Beta(1, 1)
    #         y ~ Normal(x, 1)
    #     end
    #     yval = 3

    #     betas = [0.0, 0.5, 1.0]
    #     proposal = AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(0.05), 10)
    #     anis = AnISHMC(
    #         betas,
    #         proposal,
    #         10,
    #         SimpleRejection()
    #     )

    #     #chain = sample(Random.GLOBAL_RNG, test_model(yval), anis, 10)
    #     #xs = Array(chain[:x])
    #     #@test all(0 .< xs) && all(xs .<= 1)

    #     true_logjoint(x) = logpdf(Beta(1, 1), x) + logpdf(Normal(x, 1), yval)
    #     tm = test_model(yval)
    #     spl = Turing.DynamicPPL.Sampler(anis, tm)
    #     Turing.DynamicPPL.link!(spl.state.vi, spl)
    #     logjoint = AnnealedIS.gen_logjoint(spl.state.vi, tm, spl)

    #     @test logjoint([2.0]) == true_logjoint(2)
    #     @show logjoint([2.0])
    #     @test logjoint([0.5]) == true_logjoint(0.5)
    # end

    @testset "Density closures" begin
        @model function test_model(y)
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
        end

        betas = [0.0, 0.5, 1.0]
        proposal = AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(0.05), 10)
        anis = AnISHMC{Turing.Core.ReverseDiffAD{false}}(
            betas,
            proposal,
            10,
            SimpleRejection()
        )

        yval = 3
        tm = test_model(yval)
        spl = Turing.DynamicPPL.Sampler(anis, tm)

        vi = spl.state.vi
        logjoint = AnnealedIS.gen_logjoint(vi, tm, spl)
        logprior = AnnealedIS.gen_logprior(vi, tm, spl)

        true_logjoint(x) = logpdf(Normal(0, 1), x[1]) + logpdf(Normal(x[1], 1), yval)
        @test logjoint([2.0]) == true_logjoint([2])

        true_logprior(x) = logpdf(Normal(0, 1), x[1])
        @test logprior([2.0]) == true_logprior([2])

        # Tests to check whether functions are ThreadSafe.
        num_vals = 20
        test_vals = rand(num_vals)
        test_vals = [[x] for x in test_vals]

        vi = Turing.VarInfo(tm)
        logjoint = AnnealedIS.gen_logjoint(vi, tm, spl)
        logjoint_grad = AnnealedIS.gen_logjoint_grad(vi, tm, spl)

        logprior = AnnealedIS.gen_logprior(vi, tm, spl)
        logprior_grad = AnnealedIS.gen_logprior_grad(vi, tm, spl)

        for fn in [logjoint, logjoint_grad, logprior, logprior_grad]
            vals_seq = map(fn, test_vals)
            vals_parallel = similar(vals_seq)
            Threads.@threads for i in 1:num_vals
                vals_parallel[i] = fn(test_vals[i])
            end
            @test vals_seq == vals_parallel

            vals_broadcast = fn.(test_vals)
            @test vals_seq == vals_broadcast
        end
    end

    @testset "AD backends" begin
        @model function test_model(y)
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
        end

        yval = 3
        tm = test_model(yval)

        num_vals = 20
        test_vals = rand(num_vals)
        test_vals = [[x] for x in test_vals]

        # Forward mode AD
        betas = [0.0, 0.5, 1.0]
        proposal = AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(0.05), 10)
        anis_fw = AnISHMC{Turing.ForwardDiffAD{1}}(
            betas,
            proposal,
            10,
            SimpleRejection()
        )
        spl = Turing.DynamicPPL.Sampler(anis_fw, tm)
        vi = spl.state.vi

        logjoint_grad_fw = AnnealedIS.gen_logjoint_grad(vi, tm, spl)
        logprior_grad_fw = AnnealedIS.gen_logprior_grad(vi, tm, spl)

        joint_grads_fw = logjoint_grad_fw.(test_vals)
        prior_grads_fw = logprior_grad_fw.(test_vals)

        # ReverseDiff AD
        anis_rv = AnISHMC{Turing.Core.ReverseDiffAD{false}}(
            betas,
            proposal,
            10,
            SimpleRejection()
        )
        spl = Turing.DynamicPPL.Sampler(anis_rv, tm)
        vi = spl.state.vi

        logjoint_grad_rv = AnnealedIS.gen_logjoint_grad(vi, tm, spl)
        logprior_grad_rv = AnnealedIS.gen_logprior_grad(vi, tm, spl)

        joint_grads_rv = logjoint_grad_rv.(test_vals)
        prior_grads_rv = logprior_grad_rv.(test_vals)

        @test joint_grads_fw == joint_grads_rv
        @test prior_grads_fw == prior_grads_rv

        # Some sanity check that the joint and prior are different
        @test prior_grads_rv != joint_grads_rv
        @test prior_grads_fw != joint_grads_fw
    end
end

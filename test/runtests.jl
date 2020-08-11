using Turing
using Distributions
using AdvancedMH
using LinearAlgebra: I
using Random
using Test

using AnnealedIS

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
        @test isapprox(posterior_mean, [1.5]; atol=1e-2)

        ess = effective_sample_size(samples)
        # TODO: What is a reasonable ESS to expect?
        @test ess > 900
        @test ess == diagnostics[:ess]
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
            y ~ Normal(x, 1)
        end

        y_obs = 2
        tm = test_model(y_obs)

        ais = AnnealedISSampler(tm, N)
        samples, diagnostics = ais_sample(rng, ais, num_samples)
    end

    @testset "Transition kernels" begin
        prior_sample = (a = 0.0, b = [1; 2])
        expected_transition_kernel = (a = Normal(0, 1), b = MvNormal(2, 1.0))
        transition_kernel = AnnealedIS.get_normal_transition_kernel(prior_sample, 1)
        @test expected_transition_kernel == transition_kernel
    end

    @testset "Intermediate Samples" begin
        N = 10 
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
        @test size(inter_samples) == (num_samples, 2)
        @test map(x->x.params, samples) == inter_samples[:,end]
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
end

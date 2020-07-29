using Turing
using Distributions
using AdvancedMH
using LinearAlgebra: I
using Random
using Test

using AnnealedIS

@testset "AnnealedIS.jl" begin
    # TODO: Some basic test on a problem with known solution so we know algorithm does 
    #       something reasonable.
    rng = MersenneTwister(42)
    Random.seed!(42)

    @testset "AnnealedISSampler" begin
        D = 5
        N = 2

        prior = MvNormal(D, 1.0)
        density(params) = logpdf(MvNormal(D, 0.5), params)
        joint = DensityModel(density)

        ais = AnnealedISSampler(prior, joint, N)

        @test ais.betas == [0.0, 0.5, 1.0]
        x = rand(rng, ais.prior)
        y = rand(rng, ais.transition_kernels[1](x))
        @test typeof(x) == typeof(y)

        samples = [rand(rng, ais.prior) for _ in 1:N]

        @test AnnealedIS.logdensity(ais, 1, samples[1]) == logpdf(prior, samples[1])
        @test AnnealedIS.logdensity(ais, 3, samples[1]) == density(samples[1])
        @test AnnealedIS.logdensity(ais, 2, samples[1]) == (0.5*logpdf(prior, samples[1]) + 0.5*density(samples[1]))
    end

    @testset "Convergence test" begin
        D = 1
        N = 100
        num_samples = 1000

        y_obs = [3.0]

        prior = MvNormal(D, 1.0)
        density(params) = logpdf(MvNormal(params, I), y_obs) + logpdf(prior, params)
        joint = DensityModel(density)

        ais = AnnealedISSampler(prior, joint, N)

        samples = ais_sample(rng, ais, num_samples)

        posterior_mean = AnnealedIS.estimate_expectation(samples, x -> x)
        @test isapprox(posterior_mean, [1.5]; atol=1e-2)
    end

    @testset "Sample from Turing" begin
        @model function test_model(y)
            x ~ Normal(0, 1)
            y ~ Normal(y, 1)
        end

        y_obs = 2
        tm = test_model(y_obs)

        named_tuple = sample_from_prior(rng, tm)
        @test typeof(named_tuple[:x]) == Float64

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
        @test_broken typeof(named_tuple[:x]) == typeof(rand(MvNormal(D, 1.0)))
    end

    @testset "Prior density from Turing" begin
        @model function test_model(y)
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
        end

        y_obs = 2
        tm = test_model(y_obs)
        logprior_density = make_log_prior_density(tm)

        xval = 1
        nt = (x = xval,)
        @test logprior_density(nt) == logpdf(Normal(0, 1), xval)
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
end

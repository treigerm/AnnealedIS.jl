using AnnealedIS
using Distributions
using AdvancedMH
using LinearAlgebra: I
using Test
using Random

@testset "AnnealedIS.jl" begin
    # TODO: Some basic test on a problem with known solution so we know algorithm does 
    #       something reasonable.
    rng = MersenneTwister(42)

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
        post_mean = sum(samples) do weighted_sample
            exp(weighted_sample.log_weight) * weighted_sample.params
        end
        post_mean_denominator = sum(samples) do weighted_sample
            exp(weighted_sample.log_weight)
        end

        standard_mean = sum(samples) do weighted_sample
            weighted_sample.params
        end

        @show post_mean / post_mean_denominator
    end
end

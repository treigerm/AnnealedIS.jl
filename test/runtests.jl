using AnnealedIS
using Distributions
using AdvancedMH
using Test
using Random

@testset "AnnealedIS.jl" begin
    rng = MersenneTwister(42)

    @testset "AnnealedISSampler" begin
        D = 5
        N = 3

        prior = MvNormal(D, 1.0)
        density(params) = logpdf(MvNormal(D, 1.0), params)
        joint = DensityModel(density)

        ais = AnnealedISSampler(prior, joint, N)

        @test ais.betas == [0.0, 0.5, 1.0]
        x = rand(rng, ais.prior)
        y = rand(rng, ais.transition_kernels[1](x))
        @test typeof(x) == typeof(y)

        samples = [rand(rng, ais.prior) for _ in 1:N]
        weight = AnnealedIS.weight(samples, ais)
    end

    @testset "AbstractMCMC Integration" begin
        D = 5
        N = 3

        prior = MvNormal(D, 1.0)
        density(params) = logpdf(MvNormal(D, 1.0), params)
        joint = DensityModel(density)

        ais = AnnealedISSampler(prior, joint, N)

        samples = sample(rng, joint, ais, 10)

        @show samples
    end
end

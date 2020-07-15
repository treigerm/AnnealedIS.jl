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
end

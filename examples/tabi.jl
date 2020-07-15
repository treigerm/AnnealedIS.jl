using AnnealedIS
using Distributions
using AdvancedMH
using Random

using LinearAlgebra: I
using StatsFuns: logsumexp

rng = MersenneTwister(42)

# TODO: Double check everything when we use log computations and stuff.

num_samples = 10^4

D = 10
y = 2

x_obs = - (y/sqrt(D)) * ones(D)
x_pred_obs = (y/sqrt(D)) * ones(D)

prior = MvNormal(D, 1.0)
joint_density(mu) = logpdf(MvNormal(mu, I), x_obs) + logpdf(prior, mu)
joint_model = DensityModel(joint_density)

N = 100
ais = AnnealedISSampler(prior, joint_model, N)

samples = ais_sample(rng, ais, num_samples)

log_weights = map(s -> s.log_weight, samples)
params = map(s -> s.params, samples)

true_value = pdf(MvNormal(0.5*x_obs, I), x_pred_obs)

log_denominator = logsumexp(log_weights)
numerator = logsumexp(log_weights + logpdf.(fill(MvNormal(x_pred_obs, I), length(params)), params))
estimated_value = exp(numerator - log_denominator)

relative_square_error = (estimated_value - true_value)^2 / true_value^2

@show true_value
@show estimated_value
@show relative_square_error

# TODO: Double check we have the right target.
joint_density_target(mu) = logpdf(MvNormal(mu, I), x_obs) + logpdf(prior, mu) + logpdf(MvNormal(x_pred_obs, 0.5*I), mu)
ais_target = AnnealedISSampler(prior, DensityModel(joint_density_target), N)

target_samples = ais_sample(rng, ais_target, num_samples)

target_log_weights = map(s -> s.log_weight, target_samples)
denominator_target = logsumexp(target_log_weights)

tabi_estimate = exp(denominator_target - log_denominator)

tabi_error = (tabi_estimate - true_value)^2 / true_value^2


@show tabi_estimate
@show tabi_error


# Example Julia code for Bayes' theorem manually
#
# 2025 by Ralf Herbrich
# Hasso-Plattner Institute

module Bayes

using Plots
using Distributions

struct Bernoulli
    η0::Float64
    η1::Float64
end

function Bernoulli(p::Float64)
    @assert 0.0 < p < 1.0 "Probability must be in [0,1]"
    η0 = log(1 - p) 
    η1 = log(p)
    Bernoulli(η0, η1)
end

function mean(b::Bernoulli)
    exp(b.η1) / (exp(b.η0) + exp(b.η1))
end

function Base.:*(b1::Bernoulli, b2::Bernoulli)
    Bernoulli(b1.η0 + b2.η0, b1.η1 + b2.η1)
end

function bayes(prior::Bernoulli, likelihood::Bernoulli)
    p0 = (1 - mean(prior)) * (1 - mean(likelihood))
    p1 = mean(prior) * mean(likelihood)
    normalization = p0 + p1

    return Bernoulli(p1 / normalization)
end


function compute_example()
    prior = Bernoulli(0.2)
    println("Prior mean: ", mean(prior))

    likelihood1 = Bernoulli(0.9)
    posterior1 = prior * likelihood1
    posterior1_alternative = bayes(prior, likelihood1)

    println("Likelihood mean: ", mean(likelihood1))
    println("Posterior mean: ", mean(posterior1))
    println("Posterior mean (alternative): ", mean(posterior1_alternative))

    likelihood2 = Bernoulli(0.9)
    posterior2 = posterior1 * likelihood2
    posterior2_alternative = bayes(posterior1_alternative, likelihood2)

    println("Likelihood mean: ", mean(likelihood2))
    println("Posterior mean: ", mean(posterior2))
    println("Posterior mean (alternative): ", mean(posterior2_alternative))

    likelihood3 = Bernoulli(0.9)
    posterior3 = posterior2 * likelihood3
    posterior3_alternative = bayes(posterior2_alternative, likelihood3)

    println("Likelihood mean: ", mean(likelihood3))
    println("Posterior mean: ", mean(posterior3))
    println("Posterior mean (alternative): ", mean(posterior3_alternative))
end

function plot_example(; data = [0.9, 0.6, 0.7, 0.5, 0.7, 0.8, 0.6], prior=Bernoulli(0.5))
    current = prior
    η0 = Vector{Float64}()
    η1 = Vector{Float64}()
    p0 = Vector{Float64}()
    p1 = Vector{Float64}()
    for x in data
        push!(η0, current.η0)
        push!(η1, current.η1)
        push!(p0, 1-mean(current))
        push!(p1, mean(current))
        likelihood = Bernoulli(x)
        current = current * likelihood
    end
    push!(η0, current.η0)
    push!(η1, current.η1)
    push!(p0, 1-mean(current))
    push!(p1, mean(current))

    p = plot()
    plot!(p, η0, η1, label="Posterior", xlabel="η0", ylabel="η1", legend=:topright, lw = 3, title="Natural parameters")
    scatter!(p, η0, η1, label=false, color=:red)
    display(p)

    p = plot()
    plot!(p, p0, p1, label="Posterior", xlabel="p0", ylabel="p1", legend=:topright, lw = 3, title="Mean parameters", xlim=(0,1), ylim=(0,1))
    scatter!(p, p0, p1, label=false, color=:red)
    display(p)
end

function main()
    compute_example()
    plot_example()
    plot_example(data = rand(Beta(8,2), 100), prior=Bernoulli(0.5))
end 

end
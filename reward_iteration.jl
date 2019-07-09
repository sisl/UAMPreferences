"""
reward_iteration.jl
- Implements reward iteration method for solving Markov Decision Process with Preferences
05/2019 S. M. Katz smkatz@stanford.edu
"""

using Mamba
using Plots
using Distributions
using LinearAlgebra
using JLD2
using SparseArrays
using KernelDensity

include("ri_types.jl")
include("ri_const.jl")
include("ri_functions.jl")
include("landing_mdp.jl")

Random.seed!(4)

"""
function reward_iteration
	- main function for solving MDPP
	INPUTS:
	- N: number of iterations
	- verbose: whether or not to print out information when
			   creating queries
"""
function reward_iteration(N::Int64; verbose=false)
	for i = 1:N
		sample_w(M)
		query = create_query(;verbose=verbose)
		ψ = get_ψ(query)
		pref = obtain_preference(query)
		push!(prefs, Preference(ψ, pref))
	end
	W = sample_w(M)
	return W
end
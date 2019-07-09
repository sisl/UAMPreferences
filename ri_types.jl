"""
ri_types.jl
- Defines types for MDPP implementation using reward iteration algorithm
05/2019 S. M. Katz smkatz@stanford.edu
"""

"""
type Trajectory
	- Sequence of x and y values for each aircraft
	- NOT sequence of states and actions for now
"""
struct Trajectory
	y::Array{Float64}
	z::Array{Float64}
	ẏ::Array{Float64}
	ż::Array{Float64}
	ÿ::Array{Float64}
	z̈::Array{Float64}
end

"""
type Query
	- Pair of trajectories to ask user about
"""
struct Query
	τ₁::Vector{Trajectory}
	τ₂::Vector{Trajectory}
end

"""
type Preference
	- Contains ψ, preference pairs (both used in MCMC)
	- pref is +1 if prefer 1 to 2 and -1 if prefer 2 to 1
"""
mutable struct Preference
	ψ::Vector
	pref::Int64
end

"""
type Weights
	- Contains weight samples
"""
struct Weights
	W::Array{Float64,2}
end

"""
type WeightSet
	- Contains set of weights
"""
struct WeightSet
	w₁::Vector{Float64}
	w₂::Vector{Float64}
end
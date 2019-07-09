"""
interactive_reward_iteration.jl
- Works with IJulia using Julia's interact package
05/2019 S. M. Katz smkatz@stanford.edu
"""

using Mamba
using Distributions
using LinearAlgebra
using JLD2

using PGFPlots
using Interact
include("support_code.jl")

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
"""
function reward_iteration(N::Int64; verbose=false)
	for i = 1:N
		sample_w(M)
		query = create_query(;verbose=verbose)
		ψ = get_ψ(query)
		println("ψ: $ψ")
		pref = obtain_preference(query)
		push!(prefs, Preference(ψ, pref))
	end
	W = sample_w(M)
	#solve_trajectory()
	return W
end

"""
function interactive_reward_iteration
	- will run in an interactive Jupyter notebook
	- main function for solving MDPP
"""
function interactive_reward_iteration()
	currupdate = 0
	currnew_query = 0
	@manipulate for new_query in button("Next Query"),
		update in button("Update"),
		finish in button("Finish"),
		UserPreference in ["Model 1", "Model 2"]
		if new_query > currnew_query
			println("Creating Next Query...")
			currnew_query += 1
			sample_w(M)
			query = create_query()
			curr_pref.ψ = get_ψ(query)
			plot_query_pgf(query)
		elseif update > currupdate
			currupdate += 1
			if UserPreference == "Model 1"
				curr_pref.pref = 1
			else
				curr_pref.pref = -1
			end
			push!(prefs, Preference(curr_pref.ψ, curr_pref.pref))
		elseif finish > 0
			sample_w(M)
		end
	end
end
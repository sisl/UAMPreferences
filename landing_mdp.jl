"""
landing_mdp.jl
- Implements landing mdp for uam landing application of preference based encounter modeling
05/2019 S. M. Katz smkatz@stanford.edu
"""

using LinearAlgebra
using GridInterpolations
using POMDPs
using POMDPModelTools
using DiscreteValueIteration
using SparseArrays
using JLD2

using Distributions
#using Plots

using Random
Random.seed!(18)

include("ri_types.jl")
include("ri_const.jl")

"""
----------------------------------------------
General Constants
----------------------------------------------
"""
ft2m = 0.3048
g = 9.81
fpm2mps = 0.00508

"""
----------------------------------------------
Constants - Fill in!
----------------------------------------------
"""

# Define state discretizations
altitudes = collect(range(0, stop=500*ft2m, length=50))
vertical_rates = collect(range(-500*fpm2mps, stop=0, length=4))
ground_speeds = collect(range(0, stop=60, length=15))

# Define actions (actions are joint horizontal and vertical accelerations)
a2ind = Dict(:z̈₁ÿ₁ => 1, :z̈₁ÿ₂ => 2, :z̈₁ÿ₃ => 3, :z̈₁my => 4, 
		:z̈₂ÿ₁ => 5, :z̈₂ÿ₂ => 6, :z̈₂ÿ₃ => 7, :z̈₂my => 8, 
		:z̈₃ÿ₁ => 9, :z̈₃ÿ₂ => 10, :z̈₃ÿ₃ => 11, :z̈₃my => 12, 
		:mzÿ₁ => 13, :mzÿ₂ => 14, :mzÿ₃ => 15, :mzmy => 16)

a2val = Dict(:z̈₁ÿ₁ => [0.01, -0.05g], :z̈₁ÿ₂ => [0.01, -0.1g], :z̈₁ÿ₃ => [0.01, -0.2g], :z̈₁my => [0.01, 0.0], 
		:z̈₂ÿ₁ => [0.025, -0.05g], :z̈₂ÿ₂ => [0.025, -0.1g], :z̈₂ÿ₃ => [0.025, -0.2g], :z̈₂my => [0.025, 0.0], 
		:z̈₃ÿ₁ => [0.05, -0.05g], :z̈₃ÿ₂ => [0.05, -0.1g], :z̈₃ÿ₃ => [0.05, -0.2g], :z̈₃my => [0.05, 0.0], 
		:mzÿ₁ => [0.0, -0.05g], :mzÿ₂ => [0.0, -0.1g], :mzÿ₃ => [0.0, -0.2g], :mzmy => [0.0, 0.0])

nA = length(a2ind)
prev_action = collect(1:nA)

discount_factor = 0.999

# deleted g from here !!!!!!!!

dt = 4

actions_array = [:z̈₁ÿ₁, :z̈₁ÿ₂, :z̈₁ÿ₃, :z̈₁my, :z̈₂ÿ₁, :z̈₂ÿ₂, :z̈₂ÿ₃, :z̈₂my, :z̈₃ÿ₁, :z̈₃ÿ₂, :z̈₃ÿ₃, :z̈₃my, :mzÿ₁, :mzÿ₂, :mzÿ₃, :mzmy]

@load "Transition_mat.jld2" T


"""
----------------------------------------------
Define the landing mdp and its properties
----------------------------------------------
"""
struct landingMDP <: MDP{Int64, Symbol}
	grid::RectangleGrid
    nA::Int64
    α::Float64
    β::Float64
    γ::Float64
    landing_reward::Float64
    backwards_pen::Float64
end

"""
function landing_mdp
    INPUTS:
    - α: magnitude of jerk penalty
    - β: magnitude of landing speed penalty
    - γ: magnitude of acceleration penalty
    - landing_reward: reward for landing
    - backwards_pen: magnitude of penalty for going backwards
    OUTPUTS:
    - mdp for uam landing that can be solved using POMDPs.jl
"""
function landingMDP(;α=1, β=1, γ=1, landing_reward=10000, backwards_pen=0.1)
    state_grid = RectangleGrid(altitudes, vertical_rates, ground_speeds, prev_action)
    return landingMDP(state_grid, nA, α, β, γ, landing_reward, backwards_pen)
end

"""
----------------------------------------------
Functions for POMDPs.jl
----------------------------------------------
"""

function POMDPs.states(mdp::landingMDP)
    return collect(0:length(mdp.grid))
end

function POMDPs.n_states(mdp::landingMDP)
    return length(mdp.grid)+1
end

function POMDPs.stateindex(mdp::landingMDP, state::Int64)
    return state+1
end

function POMDPs.actions(mdp::landingMDP)
    # Need to select joint actions!
    # (vertical accels positive because want to slow down descending)
    # z̈₁ - 0.01 m/s²
    # z̈₂ - 0.025 m/s²
    # z̈₃ - 0.05 m/s²
    # mz - 0 
    # ÿ₁ - -0.05g
    # ÿ₂ - -0.1g
    # ÿ₃ - -0.2g
    # my - 0
    return actions_array
end

function POMDPs.n_actions(mdp::landingMDP)
    return mdp.nA
end

function POMDPs.actionindex(mdp::landingMDP, a::Symbol)
    return a2ind[a]
end

function POMDPs.discount(mdp::landingMDP)
    return discount_factor
end

function POMDPs.transition(mdp::landingMDP, s::Int64, a::Symbol)
    if s == 0 # in terminal state
        return SparseCat(0, 0)
    else
        znext, żnext, ẏnext = next_state_vals(mdp, s, a)
        prev_actionnext = a2ind[a]

        states, probs = interpolants(mdp.grid, [znext, żnext, ẏnext, prev_actionnext])

        if znext ≤ 0 # Transition to terminal state (0) (has landed)
            states = 0
            probs = 1
        end
        # Create sparse categorical distribution from this
        returnDist = SparseCat(states, probs)
        return returnDist
    end
end

function POMDPs.reward(mdp::landingMDP, s::Int64, a::Symbol)
	s_grid = ind2x(mdp.grid, s)
	z̈, ÿ = a2val[a][1], a2val[a][2]
	z, ż, ẏ = s_grid[1], s_grid[2], s_grid[3]
	prev_action_ind = convert(Int64, s_grid[4])
	prev_action = actions_array[prev_action_ind]

	znext, żnext, ẏnext = next_state_vals(mdp, s, a)

    # Initialize reward to 0
    r = 0.0

    # Incentivize landing
    if znext ≤ 0.0
    	r += mdp.landing_reward
    end

 	# Penalize acceleration
 	r -= 2*mdp.γ*norm([z̈, ÿ])

    # Penalize jerk
    r -= 20*mdp.α*norm([z̈, ÿ] - a2val[prev_action])/dt

    # Penalize speed at landing
    if 0 < z < 50*ft2m
    	r -= 3*mdp.β*min(norm([ż, ẏ])/z, 1000) # CHANGE THIS!!!!
    end

    # Penalize going backwards
    r -= mdp.backwards_pen*max(0, -ẏnext)

    return r
end

function POMDPs.isterminal(mdp::landingMDP, s::Int64)
    if s == 0
        return true
    else
        return false
    end
end

"""
----------------------------------------------
Supplementary Functions
----------------------------------------------
"""
function next_state_vals(mdp::landingMDP, s::Int64, a::Symbol)
	# First, convert to a grid state
    s_grid = ind2x(mdp.grid, s)
    # Convert action to acceleration
    z̈, ÿ = a2val[a][1], a2val[a][2]
    z, ż, ẏ = s_grid[1], s_grid[2], s_grid[3]

    znext = z + ż*dt
    żnext = ż + z̈*dt
    ẏnext = ẏ + ÿ*dt

    return znext, żnext, ẏnext
end

"""
----------------------------------------------
Solve
----------------------------------------------
"""
"""
function solve_landing_mdp
    - solves mdp using value iteration from POMDPs.jl
    INPUTS:
    - α: magnitude of jerk penalty
    - β: magnitude of landing speed penalty
    - γ: magnitude of acceleration penalty
    - landing_reward: reward for landing
    - verbose: boolean whether or not to show output from value iteration solver
    - init_util: whether or not initial utility values will be specified
"""
function solve_landing_mdp(;α=1, β=1, γ=1, landing_reward=10000, verbose=false, init_util=false)
	mdp = landingMDP(α=α, β=β, γ=γ, landing_reward=landing_reward)
	if init_util
		solver = ValueIterationSolver(verbose=verbose, init_util=util)
	else
		solver = ValueIterationSolver(verbose=verbose)
	end
	policy = solve(solver, mdp)
	return mdp, policy
end

"""
function solve_landing_mdp
    - solves mdp using sparse value iteration from POMDPs.jl (modified slightly so that it does
    not recalculate T every time it is solved - only recalculates R)
    INPUTS:
    - α: magnitude of jerk penalty
    - β: magnitude of landing speed penalty
    - γ: magnitude of acceleration penalty
    - landing_reward: reward for landing
    - verbose: boolean whether or not to show output from value iteration solver
    - init_util: whether or not initial utility values will be specified
"""
function solve_landing_mdp_sparse(;α=1, β=1, γ=1, landing_reward=10000, verbose=false, init_util=false)
	mdp = landingMDP(α=α, β=β, γ=γ, landing_reward=landing_reward)
	if init_util
		solver = SparseValueIterationSolver(verbose=verbose, init_util=util)
	else
		solver = SparseValueIterationSolver(verbose=verbose, init_T=T)
	end
	policy = solve(solver, mdp)
	return mdp, policy
end

"""
- This is taken straight from the DiscreteValueIteration package
- Use it to build T so that I do not have to redo it every time
I solve the mdp
"""
function transition_matrix_a_s_sp(mdp::MDP)
    # Thanks to zach
    na = n_actions(mdp)
    ns = n_states(mdp)
    transmat_row_A = [Float64[] for _ in 1:n_actions(mdp)]
    transmat_col_A = [Float64[] for _ in 1:n_actions(mdp)]
    transmat_data_A = [Float64[] for _ in 1:n_actions(mdp)]

    for s in states(mdp)
        si = stateindex(mdp, s)
        for a in actions(mdp, s)
            ai = actionindex(mdp, a)
            if !isterminal(mdp, s) # if terminal, the transition probabilities are all just zero
                td = transition(mdp, s, a)
                for (sp, p) in weighted_iterator(td)
                    if p > 0.0
                        spi = stateindex(mdp, sp)
                        push!(transmat_row_A[ai], si)
                        push!(transmat_col_A[ai], spi)
                        push!(transmat_data_A[ai], p)
                    end
                end
            end
        end
    end
    transmats_A_S_S2 = [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], n_states(mdp), n_states(mdp)) for a in 1:n_actions(mdp)]
    # Note: not valid for terminal states
    # @assert all(all(sum(transmats_A_S_S2[a], dims=2) .≈ ones(n_states(mdp))) for a in 1:n_actions(mdp)) "Transition probabilities must sum to 1"
    return transmats_A_S_S2
end

"""
----------------------------------------------
Generate trajectory using the policy
----------------------------------------------
"""
"""
function generate_trajectory
    INPUTS:
    - mdp: mdp to generate trajectories from
    - optimal policy for the mdp (need to solve it!)
    - init_state: initial state for the trajectory
    - num_steps: number of steps in the trajectory
    - dt: time step
    OUTPUTS:
    - trajectory resulting from rollout
"""
function generate_trajectory(mdp, policy, init_state; num_steps=60, dt=1)
	# Convert initial state to grid state
	states, probs = interpolants(mdp.grid, init_state)
	s = states[argmax(probs)]

	z = zeros(num_steps)
	y = zeros(num_steps)
	ż = zeros(num_steps)
	ẏ = zeros(num_steps)
	z̈ = zeros(num_steps)
	ÿ = zeros(num_steps)
	z[1], ż[1], ẏ[1] = init_state[1], init_state[2], init_state[3]

	for i = 2:num_steps
		a = action(policy, s)
		z̈[i], ÿ[i] = a2val[a][1], a2val[a][2]

		ż[i] = ż[i-1] + z̈[i]*dt
		ẏ[i] = ẏ[i-1] + ÿ[i]*dt

		z[i] = z[i-1] + ż[i-1]*dt + 0.5*z̈[i]*dt^2
		if z[i] < 0
			y[i:end] = (y[i-1] + ẏ[i-1]*dt + 0.5*ÿ[i]*dt^2)*ones(num_steps-i+1)
			break
		end
		y[i] = y[i-1] + ẏ[i-1]*dt + 0.5*ÿ[i]*dt^2

		states, probs = interpolants(mdp.grid, [z[i], ż[i], ẏ[i]])
		s = states[argmax(probs)]
	end
	return Trajectory(y, z, ẏ, ż, ÿ, z̈)
end

"""
function plot_τset
    INPUTS:
    - τset: set of trajectories to plot
    OUTPUTS:
    - p: plot object
"""
function plot_τset(τset)
	minz = 0

	miny, maxz, maxy = 0, 0, 0
	for i = 1:length(τset)
		miny = minimum(τset[i].y) > miny ? minimum(τset[i].y) : miny
		maxz = maximum(τset[i].z) > maxz ? maximum(τset[i].z) : maxz
		maxy = maximum(τset[i].y) > maxy ? maximum(τset[i].y) : maxy
	end

	p = Plots.plot(xlabel="East (m)", ylabel="Altitude (m)", title="Landing 1", legend=false, aspect_ratio=:equal, size=(1200,600), xlims=(miny,maxy+50), ylims=(minz,maxz+10))
	for i = 1:length(τset)
		Plots.plot!(p, τset[i].y, τset[i].z, linewidth=4)
		Plots.plot!(p, [τset[i].y[1:20:end]], [τset[i].z[1:20:end]], seriestype=:scatter, markersize=3)
	end
	return p
end

"""
function generate_trajectory_set
    INPUTS:
    - init_states: initial states for trajectories in the set
    - w: vector of weights to generate trajectory for
    - dt: time step
    OUTPUTS:
    - τset: set of trajectories from rollouts of the policy obtained from w
"""
function generate_trajectory_set(init_states, w; dt=0.1)
	mdp, policy = solve_landing_mdp_sparse(α=w[1], β=w[2], γ=w[3], init_util=false, verbose=false) # TODO: Add. init_util capabilitstate
	τset = Vector{Trajectory}()
	for i = 1:length(init_states[1])
		init_state = [init_states[j][i] for j in 1:length(init_states)]
		push!(τset, generate_trajectory(mdp, policy, init_state, num_steps=k*15, dt=dt))
	end
	return τset
end

# Function for just solving everything at once and generating trajectory plots
"""
function solve_mdp
    INPUTS:
    - α: magnitude of jerk penalty
    - β: magnitude of landing speed penalty
    - γ: magnitude of acceleration penalty
    - num_trajectories: number of trajectories to generate from mdp
    - init_states: initial states for the trajectories
                  (initial states are chosen randomly if init_states=[])
    OUTPUTS:
    - init_states: the initial states used to generate the trajectories
                  (if init_states were specified as an input, it will just return these)
    - τset: set of trajectories from rollouts of the policy obtained from the
            specified reward function parameters (α, β, γ)
"""
function solve_mdp(α, β, γ; num_trajectories=5, init_states=[])
	if init_states == []
		init_states = rand.(Uniform.(lb_s₀, ub_s₀), num_trajectories)
	end
	start = time()
	τset = generate_trajectory_set(init_states, [α, β, γ])
	println("Elapsed time: $(round(time() - start, digits=2))")
	p = plot_τset(τset)
	display(p)
	println("Φ: $(get_Φ(τset))")
	return init_states, τset
end
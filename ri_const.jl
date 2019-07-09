"""
ri_const.jl
- Defines constants for mdpp implementation
05/2019 S.M. Katz smkatz@stanford.edu
"""

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
General Size Parameters
----------------------------------------------
"""
M = 50 # Number of weights to sample to estimate the objective function
num_features = 3
k = 60 # Trajectory length

"""
----------------------------------------------
Variables Updated During Iterations
----------------------------------------------
"""
W = zeros(num_features, M) # Current samples of W (num_features x M)
prefs = Vector{Preference}() # Initialize empty preference set

"""
----------------------------------------------
MCMC Parameters
----------------------------------------------
"""
burn = 5000 # Number of samples for MCMC burn in period
Th = 100 # Thin samples by this after burn in

# Changing prior since constraining to positive and sum to one!!
# (this provides uniform samples over the probability simplex)
prior = Dirichlet(ones(num_features))

"""
----------------------------------------------
Query Selection Parameters
----------------------------------------------
"""

dt = 1

# Initial states
lb_z₀ = 300*ft2m
ub_z₀ = 500*ft2m
lb_ẏ₀ = 30
ub_ẏ₀ = 60
lb_ż₀ = -500*fpm2mps
ub_ż₀ = -300*fpm2mps

lb_s₀ = [lb_z₀, lb_ż₀, lb_ẏ₀]
ub_s₀ = [ub_z₀, ub_ż₀, ub_ẏ₀]

# Used in probabilistic q-eval
num_candidates = 5
# Number of trajectories for each query
num_trajectories = 5
# dt used when performing rollout
dt_traj = 0.1
# Tradeoff parameter for multi-objective optimization
μ = 500.0

"""
----------------------------------------------
For Interactive Part (in jupyter notebook)
----------------------------------------------
"""
curr_pref = Preference(zeros(num_features), 1)
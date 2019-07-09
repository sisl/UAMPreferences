"""
ri_functions.jl
- Defines the functions called in the reward iteration method of solving
preference-based MDPs
05/2019 S.M. Katz smkatz@stanford.edu
"""

"""
----------------------------------------------
MCMC Sampling
----------------------------------------------
"""

"""
function sample_w
	- generates samples from p(w) based on current set of preferences obtained
	INPUTS:
	- M: number of samples to generate
	OUTPUTS:
	- W: samples of weights returned as matrix with each column as one sample (num_features x M)
"""
function sample_w(M::Int64)
	# If we don't have any information yet, sample a random w based on prior
	if isempty(prefs)
		w_samples = rand(prior, M)
		W[:,:] = w_samples
	else
		# Setup
		w₀ = ones(num_features)./num_features # Just start with all components of w being equal
		Σ = Matrix{Float64}(I, num_features, num_features) ./ 10000 # For proposal distribution
		w = AMMVariate(w₀, Σ, logf)
		# Perforn the sampling
		for i = 1:(M*Th+burn)
			sample!(w, adapt = (i <= burn))
			if i > burn && mod(i,Th) == 0 # Takes into account burn in and thinning
				ind = convert(Int64, (i-burn)/Th)
				W[:,ind] = w[:]
			end
		end
	end
	for i = 1:M
		W[:,i] = W[:,i]./sum(W[:,i]) # Normalize
	end
	return W
end

"""
function logf
	- computes the log of the posterior p(w)
	- used by MCMC sampler
	- ψ is difference between reward features of τsets
	  (used to get relative reward)
	INPUTS:
	- w: vector of weights to find the log probability of
	OUTPUTS:
	- logf: log of the posterior
"""
function logf(w::DenseVector)
	if sum(w) ≥ 1.0 || any(w .< 0.0) # no samples outside probability simplex
	 	return -Inf
	else
		sum_over_prefs = 0
		for i = 1:length(prefs)
			# Sigmoid likelihood
			sum_over_prefs += log(1/(1+exp(-prefs[i].pref*w'*prefs[i].ψ)))
		end
		return sum_over_prefs
	end
end

"""
----------------------------------------------
Active Query Selection
----------------------------------------------
"""
"""
function create_query
	- main function for creating queries
	INPUTS:
	- verbose: boolean whether or not to print information
	- multiobj: whether or not to use multiobjective queying method
	            (defaults to probabilistic q-eval)
    - random: whether or not to select random queries
              (defaults to probabilistic q-eval)
    - μ: hyperparameter for multiobjective optimization method
         (ignored if multiobj=false)
     OUTPUTS:
     - query consisting of two trajectory sets
"""
function create_query(;verbose=false, multiobj=false, random=false, μ=500.0)
	###################################################################
	# Select weights and initial states
	###################################################################
	if verbose
		println("Creating Query:")
	end
	if multiobj
		query_w = choose_query_opt(μ=μ)
	elseif random
		w_sample₁ =  rand(prior, num_features)
		w_sample₂ =  rand(prior, num_features)
		w_sample₁ ./= sum(w_sample₁)
		w_sample₂ ./= sum(w_sample₂)
		query_w = WeightSet(w_sample₁, w_sample₂)
	else # probabilistic q-eval
		c = get_center()
		if verbose
			println("Selecting top candidates...")
		end
		candidates = get_top_candidates(c)
		query_w = choose_pair(candidates)
	end
	println(query_w)
	init_states = rand.(Uniform.(lb_s₀, ub_s₀), num_trajectories) # array of initial states
	###################################################################
	# Solve mdp and perform rollouts
	###################################################################
	if verbose
		println("Query selected!")
		println("Solving for first trajectory set...")
	end
	τ₁set = generate_trajectory_set(init_states, query_w.w₁)
	if verbose
		println("Solving for second...")
	end
	τ₂set = generate_trajectory_set(init_states, query_w.w₂, dt=dt_traj)
	if verbose
		println("Done!")
	end
	return Query(τ₁set, τ₂set)
end

"""
function get_center
	- used for probabilistic q-eval (just returns the mean of the samples)
	INPUTS:
	OUTPUTS:
	- center of approximated permissible region
"""
function get_center()
	c = [mean(W[i,:]) for i in 1:num_features]
	return c
end

"""
function get_top_candidates
	- used for probabilistic q-eval to select candidates with bisecting
	hyperplane closest to center of permissible region
	INPUTS:
	- c: center of permissible region (from get_center)
	OUTPUTS:
	- top candidate weight sets
"""
function get_top_candidates(c)
	candidates = Vector{WeightSet}()
	vals = Vector{Float64}()
	for i = 1:M
		for j = i+1:M # This should give all pairs
			w₁, w₂ = W[:,i], W[:,j]
			push!(candidates, WeightSet(w₁, w₂))
			# Compute distance to center
			v = w₁ - w₂
			normv = norm(v)
			mp = (w₁ + w₂)./2
			D = abs(v'*(c - mp))/norm(v)
			push!(vals, D)
		end
	end
	# Sort and choose top
	sorted_candidates = candidates[sortperm(vals)]
	return sorted_candidates[1:num_candidates]
end

"""
function choose_pair
	- used for probabilistic q-eval to choose a pair from the
	candidates that has the ratio of samples on either side of
	their bisecting hyperplane closest to one
	INPUTS:
	- candidates: vector of WeightSet (from get_top_candidates)
	OUTPUTS:
	- best WeightSet among the candidates
"""
function choose_pair(candidates)
	fracs = Vector{Float64}()
	for i = 1:length(candidates)
		w₁ = candidates[i].w₁
		w₂ = candidates[i].w₂
		mp = (w₁ + w₂)/2
		n̂ = (w₁ - w₂)./norm(w₁ - w₂)
		prods = W'*n̂
		push!(fracs, length(findall(prods .> n̂'*mp))/M)
	end
	ind = argmin(abs.(fracs .- 0.5)) # Choose fraction closest to even
	return candidates[ind]
end

"""
function choose_query_opt
	- choose query samples based on multiobjective optimization
	- balances between choosing samples that are different but
	also likely
	INPUTS:
	- μ: parameter that balances between likelihood and distance
	    (higher μ means more weight on distance)
"""
function choose_query_opt(;μ=2)
	best_val = 0
	best_candidate = WeightSet(W[:,1], W[:,2])
	# Get kernel density estimate for estimating likelihood
	k = kde(W[1:2,:]')
	ik = InterpKDE(k)
	# Loop through candidates (MCMC samples) and find the best
	for i = 1:M
		for j = i+1:M
			pa = pdf(ik, W[1,i], W[2,i])
			pb = pdf(ik, W[1,j], W[2,j])
			distance = norm(W[:,i] - W[:,j])
			if pa*pb + μ*distance > best_val
				best_val = pa*pb + μ*distance
				best_candidate = WeightSet(W[:,i], W[:,j])
			end
		end
	end
	return best_candidate
end

"""
function get_ψ
	- get reward feature difference between trajectory sets in query 
	  (used to evaluate differences in reward)
	INPUTS:
	- query: query to obtain ψ for
	OUTPUTS:
	- ψ: features of trajectory set 1 - features of trajectory set 2
"""
function get_ψ(query)
	return get_Φ(query.τ₁) - get_Φ(query.τ₂)
end

"""
function get_ϕ
	- get features of a particular trajectory set (to evaluate reward)
	- NOTE: the way the reward function is defined in this application
	allows this type of implementation (specifically, the reward parameters
	can simply be multipled by certain trajectory features in order to
	determine the total reward over a trajectory)
	INPUTS:
	- query: query to obtain ψ for
	OUTPUTS:
	- ψ: features of trajectory set 1 - features of trajectory set 2
"""
function get_Φ(τset)
	Φ = zeros(num_features)
	airtime = 0 # Time in air (for averaging)
	neartime = 0 # Time in speed penalty region near the ground (for averaging)
	for i = 1:length(τset)
		airtime += length(findall(τset[i].z .> 0))
		neartime += length(findall(0 .< τset[i].z .< 50*ft2m))
		z̈ = τset[i].z̈
		ÿ = τset[i].ÿ
		jerk = 20*[norm([z̈[j+1], ÿ[j+1]] - [z̈[j], ÿ[j]]) for j=1:length(z̈)-1]./dt_traj
		Φ[1] -= sum(jerk)
		for j = 1:length(τset[1].y)
			if 0 < τset[i].z[j] < 50*ft2m
		    	Φ[2] -= (1/5)*min(norm([τset[i].ż[j], τset[i].ẏ[j]])/τset[i].z[j], 1000)
		    end
		end
		accel = 2*[norm([z̈[j], ÿ[j]]) for j = 1:length(z̈)]
		Φ[3] -= sum(accel)
	end
	Φ[1] = Φ[1]./airtime
	Φ[2] = Φ[2]./neartime
	Φ[3] = Φ[3]./airtime
	return Φ
end

"""
----------------------------------------------
Obtain Preference
----------------------------------------------
"""

"""
function obtain_preference
	- Get the preference of the user
	INPUTS:
	- query: query to ask the user about
	OUTPUTS:
	- pref: +1 if prefer 1 to 2 and -1 if prefer 2 to 1 
"""
function obtain_preference(query::Query)
	# Plot them for the user
	p = plot_query(query)
	display(p)

	# Ask the user for their preference
	resp = Input("Which trajectory set is more realistic? (1/2): ")

	if resp == "1"
		pref = 1
	else
		pref = -1
	end

	return pref
end

"""
function Input
	- This function will be used to get user responses to preference queries
	INPUTS:
	- prompt: What to ask the user for
	OUTPUTS:
	- user input
"""
function Input(prompt::String)
    print(prompt)
    return readline()
end

"""
function plot_query
	- plot query to show to expert using plots.jl
	INPUTS:
	- query: query to plot
	OUTPUTS:
	- p: the plot object to display
"""
function plot_query(query)
	minz = 0

	miny, maxz, maxy = 0, 0, 0
	for i = 1:length(query.τ₁)
		miny = minimum([query.τ₁[i].y; query.τ₂[i].y]) > miny ? minimum([query.τ₁[i].y; query.τ₂[i].y]) : miny
		maxz = maximum([query.τ₁[i].z; query.τ₂[i].z]) > maxz ? maximum([query.τ₁[i].z; query.τ₂[i].z]) : maxz
		maxy = maximum([query.τ₁[i].y; query.τ₂[i].y]) > maxy ? maximum([query.τ₁[i].y; query.τ₂[i].y]) : maxy
	end

	# Create plot objects
	p₁ = Plots.plot(xlabel="East (m)", ylabel="Altitude (m)", title="Landing 1", legend=false, aspect_ratio=:equal, size=(1200,600), xlims=(miny,maxy+50), ylims=(minz,maxz+10))
	for i = 1:length(query.τ₁)
		Plots.plot!(p₁, query.τ₁[i].y, query.τ₁[i].z, linewidth=4)
		Plots.plot!(p₁, [query.τ₁[i].y[1:20:end]], [query.τ₁[i].z[1:20:end]], seriestype=:scatter, markersize=3)
	end

	p₂ = Plots.plot(xlabel="East (m)", ylabel="Altitude (m)", title="Landing 2", legend=false, aspect_ratio=:equal, size=(1200,600), xlims=(miny,maxy+50), ylims=(minz,maxz+10))
	for i = 1:length(query.τ₂)
		Plots.plot!(p₂, query.τ₂[i].y, query.τ₂[i].z, linewidth=4)
		Plots.plot!(p₂, [query.τ₂[i].y[1:20:end]], [query.τ₂[i].z[1:20:end]], seriestype=:scatter, markersize=3)
	end

	p = Plots.plot(p₁, p₂, layout=(2,1))

	return p
end

"""
function plot_query_pgf
	- plot query to show to expert using PGFPlots 
	(for interactive Jupyter notebook version)
	INPUTS:
	- query: query to plot
	OUTPUTS:
	- g: the group plot to display
"""
function plot_query_pgf(query)
	zmin = 0

	ymin, zmax, ymax = 0, 0, 0
	for i = 1:length(query.τ₁)
		ymin = minimum([query.τ₁[i].y; query.τ₂[i].y]) > ymin ? minimum([query.τ₁[i].y; query.τ₂[i].y]) : ymin
		zmax = maximum([query.τ₁[i].z; query.τ₂[i].z]) > zmax ? maximum([query.τ₁[i].z; query.τ₂[i].z]) : zmax
		ymax = maximum([query.τ₁[i].y; query.τ₂[i].y]) > ymax ? maximum([query.τ₁[i].y; query.τ₂[i].y]) : ymax
	end

	# Create plot objects
	ax₁ = Axis(height="20cm", width="20cm", axisEqualImage=true, xmin=ymin, xmax=ymax, ymin=zmin, ymax=zmax)
	ax₁.xlabel = "East (m)"
	ax₁.ylabel = "Altitude (m)"
	ax₁.title = "Policy 1"
	for i = 1:length(query.τ₁)
		push!(ax₁, Plots.Linear(query.τ₁[i].y[1:20:end], query.τ₁[i].z[1:20:end], markSize=1, mark=*))
	end

	ax₂ = Axis(height="20cm", width="20cm", axisEqualImage=true, xmin=ymin, xmax=ymax, ymin=zmin, ymax=zmax)
	ax₂.xlabel = "East (m)"
	ax₂.ylabel = "Altitude (m)"
	ax₂.title = "Policy 2"
	for i = 1:length(query.τ₂)
		push!(ax₂, Plots.Linear(query.τ₂[i].y[1:20:end], query.τ₂[i].z[1:20:end], markSize=1, mark=*))
	end

	g = GroupPlot(1, 2, groupStyle = "vertical sep = 2cm")
	push!(g, ax₁)
	push!(g, ax₂)

	return g
end

"""
function plot_query_speed_pgf
	- plot query to show to expert using PGFPlots with speed plots on side
	(for interactive Jupyter notebook version)
	INPUTS:
	- query: query to plot
	OUTPUTS:
	- g: the group plot to display
"""
function plot_query_speed_pgf(query)
	zmin = 0

	ymin, zmax, ymax = 0, 0, 0
	for i = 1:length(query.τ₁)
		ymin = minimum([query.τ₁[i].y; query.τ₂[i].y]) > ymin ? minimum([query.τ₁[i].y; query.τ₂[i].y]) : ymin
		zmax = maximum([query.τ₁[i].z; query.τ₂[i].z]) > zmax ? maximum([query.τ₁[i].z; query.τ₂[i].z]) : zmax
		ymax = maximum([query.τ₁[i].y; query.τ₂[i].y]) > ymax ? maximum([query.τ₁[i].y; query.τ₂[i].y]) : ymax
	end

	# Create plot objects
	ax₁ = Axis(height="20cm", width="20cm", axisEqualImage=true, xmin=ymin, xmax=ymax, ymin=zmin, ymax=zmax)
	ax₁.xlabel = "East (m)"
	ax₁.ylabel = "Altitude (m)"
	ax₁.title = "Policy 1"
	ax₁s = Axis(height="5cm", width="5cm")
	ax₁s.xlabel = "Time (s)"
	ax₁s.ylabel = "Speed (m/s)"
	ax₁s.title = "Policy 1"
	for i = 1:length(query.τ₁)
		push!(ax₁, Plots.Linear(query.τ₁[i].y[1:20:end], query.τ₁[i].z[1:20:end], markSize=1, mark=*))
		
		times = collect(range(0, step=dt_traj, length=length(query.τ₁[i].y)-1))
		ẏ = [(query.τ₁[i].y[j+1] - query.τ₁[i].y[j])/dt_traj for j=1:length(query.τ₁[i].y)-1]
		ż = [(query.τ₁[i].z[j+1] - query.τ₁[i].z[j])/dt_traj for j=1:length(query.τ₁[i].z)-1]
		speeds = [norm([ẏ[j], ż[j]]) for j=1:length(query.τ₁[i].y)-1]
		push!(ax₁s, Plots.Linear(times, speeds, mark="none"))
	end

	ax₂ = Axis(height="20cm", width="20cm", axisEqualImage=true, xmin=ymin, xmax=ymax, ymin=zmin, ymax=zmax)
	ax₂.xlabel = "East (m)"
	ax₂.ylabel = "Altitude (m)"
	ax₂.title = "Policy 2"
	ax₂s = Axis(height="5cm", width="5cm")
	ax₂s.xlabel = "Time (s)"
	ax₂s.ylabel = "Speed (m/s)"
	ax₂s.title = "Policy 2"
	for i = 1:length(query.τ₂)
		push!(ax₂, Plots.Linear(query.τ₂[i].y[1:20:end], query.τ₂[i].z[1:20:end], markSize=1, mark=*))
		
		times = collect(range(0, step=dt_traj, length=length(query.τ₂[i].y)-1))
		ẏ = [(query.τ₂[i].y[j+1] - query.τ₂[i].y[j])/dt_traj for j=1:length(query.τ₂[i].y)-1]
		ż = [(query.τ₂[i].z[j+1] - query.τ₂[i].z[j])/dt_traj for j=1:length(query.τ₂[i].z)-1]
		speeds = [norm([ẏ[j], ż[j]]) for j=1:length(query.τ₂[i].y)-1]
		push!(ax₂s, Plots.Linear(times, speeds, mark="none"))
	end

	g = GroupPlot(2, 2, groupStyle = "vertical sep = 2cm")
	push!(g, ax₁)
	push!(g, ax₁s)
	push!(g, ax₂)
	push!(g, ax₂s)

	return g
end
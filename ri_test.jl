"""
ri_test.jl
- Test MDPP by specifying a true reward function and answering queries based on
which trajectory gets a higher reward
"""

using Random
Random.seed!(18)

w_true = [0.1, 0.8, 0.1]
w_true = w_true./sum(w_true)

"""
function auto_reward_iteration
	- answers queries automatically according to some prespecified reward function
	INPUTS:
	- N: number of queries to answer
	- μ: parameter for multiobjective optimization querying method
	    (ignored if multiobj=false in create_query)
	- err_rate: rate at which to answer the queries inconsistently according
	            to true reward function
    OUTPUTS:
    - W: final weight samples
"""
function auto_reward_iteration(N::Int64; μ=μ, err_rate=0)
	for i = 1:N
		start = time()
		sample_w_store(M)
		query = create_query(verbose=false, multiobj=true, random=false, μ=500.0) # Note: changed to hard code!
		ψ = get_ψ(query)

		Φ₁ = get_Φ(query.τ₁)
		Φ₂ = get_Φ(query.τ₂)

		
		if w_true'*Φ₁ > w_true'*Φ₂
			if rand() > err_rate
				pref = 1 # answer correctly
			else
				pref = -1 # err
			end
		else
			if rand() > err_rate
				pref = -1 # answer correctly
			else
				pref = 1 # err
			end
		end

		push!(prefs, Preference(ψ, pref))
		println("Iter: $i, Elapsed time: $(time()-start)")
		push!(Query_hist, query)
	end
	W = sample_w_store(M)
	return W
end

"""
function sample_w_store
	- samples weights using MCMC (adaptive metropolis) and stores them
	INPUTS:
	- M: number of samples to generate
	OUTPUTS:
	- W: samples of weights returned as matrix with each column as one sample
"""
function sample_w_store(M::Int64)
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
	push!(w_hist, Weights(copy(W))) # Store stuff
	push!(W_hist, Weights(copy(W))) # Store stuff
	return W
end

"""
function post_process_w_hist
	- post process auto_preference_iteration to obtain weights and cosine similarity, m, at each step
	INPUTS:
	- w_hist: vector of Weights sampled for each iteration (probably want w_hist[2:end])
	- w_true: true weight vector used to compute m
	OUTPUTS:
	- m: measure of how close w samples are to true weights (cosine similarity)
	- w_mean_hist: history of mean w
"""
function post_process_w_hist(w_hist::Vector, w_true::Vector)
	num_iter = length(w_hist)
	num_features = length(w_true)
	w_mean_hist = zeros(num_iter, num_features)
	m = zeros(num_iter)

	for i = 1:num_iter
		W = w_hist[i].W
		w_mean = [mean(W[i,:]) for i in 1:num_features]
		w_mean /= sum(w_mean) # Normalize w_mean
		w_mean_hist[i,:] = w_mean
		m[i] = w_mean'*w_true/(norm(w_mean)*norm(w_true))
	end

	return m, w_mean_hist
end

# Random.seed!(18)

# num_iter = 80
# num_trials = 5

# # μ_mat = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
# # μ_mat = [100.0, 500.0]
# μ_mat = [1000.0, 5000.0]

# m_mat_tot = zeros(num_iter+1,num_trials,length(μ_mat))
# W_hist = w_hist = Vector{Weights}()
# Query_hist = Vector{Query}()

# for j = 1:length(μ_mat)
# 	Random.seed!(18)
# 	m_mat = zeros(num_iter+1, num_trials)
# 	for i = 1:num_trials
# 		@eval begin
# 			w_hist = Vector{Weights}()
# 			W = zeros(num_features, M)
# 			prefs = Vector{Preference}()
# 			w = zeros(num_features)
# 		end

# 		auto_reward_iteration(num_iter, μ=μ_mat[j])

# 		# m, w_mean_hist = post_process_w_hist(w_hist[2:end], w_true)
# 		m, w_mean_hist = post_process_w_hist(w_hist, w_true)
# 		m_mat_tot[:,i,j] = m
# 		m_mat[:,i] = m

# 		println("Iter: $i")
# 	end
# 	filename = "musweep$(μ_mat[j]).jld2"
# 	println(filename)
# 	@save filename m_mat w_hist
# end

# @save "mu_sweep_extradata_10005000.jld2" m_mat_tot W_hist

 num_iter = 80
 num_trials = 1

 m_mat = zeros(num_iter+1,num_trials)
 W_hist = w_hist = Vector{Weights}()
 Query_hist = Vector{Query}()

 Random.seed!(18)
 for i = 1:num_trials
 	@eval begin
 		w_hist = Vector{Weights}()
 		W = zeros(num_features, M)
 		prefs = Vector{Preference}()
 		w = zeros(num_features)
 	end

 	auto_reward_iteration(num_iter)

 	# m, w_mean_hist = post_process_w_hist(w_hist[2:end], w_true)
 	m, w_mean_hist = post_process_w_hist(w_hist, w_true)
 	m_mat[:,i] = m

 	println("Iter: $i")
end

# @save "multiobj_dir_mu500_use.jld2" m_mat W_hist Query_hist

# Random.seed!(18)

# num_iter = 80
# num_trials = 5

# err_mat = [0.1, 0.2, 0.3]

# m_mat_tot = zeros(num_iter+1,num_trials,length(err_mat))
# W_hist = w_hist = Vector{Weights}()
# Query_hist = Vector{Query}()

# for j = 1:length(err_mat)
# 	Random.seed!(18)
# 	m_mat = zeros(num_iter+1, num_trials)
# 	for i = 1:num_trials
# 		@eval begin
# 			w_hist = Vector{Weights}()
# 			W = zeros(num_features, M)
# 			prefs = Vector{Preference}()
# 			w = zeros(num_features)
# 		end

# 		auto_reward_iteration(num_iter, μ=500.0, err_rate=err_mat[j])

# 		# m, w_mean_hist = post_process_w_hist(w_hist[2:end], w_true)
# 		m, w_mean_hist = post_process_w_hist(w_hist, w_true)
# 		m_mat_tot[:,i,j] = m
# 		m_mat[:,i] = m

# 		println("Iter: $i")
# 	end
# 	filename = "errsweepmultiobj$(err_mat[j]).jld2"
# 	println(filename)
# 	@save filename m_mat w_hist
# end

# @save "errsweep_multiobj_mu500_data.jld2" m_mat_tot W_hist

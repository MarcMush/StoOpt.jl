# developed with Julia 1.0.3
#
# functions for Stochastic Dynamic Programming 


#function admissible_state(x::Array{Float64}, states::Grid)
#	"""check if x is in states: return a boolean
#
#	x > state point
#	states > discretized state space
#
#	"""
#
#	for i in 1:length(x)
#
#		if x[i] < states[i][1]
#			return false
#		elseif x[i] > states[i][end]
#			return false
#		end
#
#	end
#
#	return true
#
#end

function compute_value_functions(train_noises::Union{Noise, Array{Noise}}, 
	controls::Grid, states::Grid, dynamics::Function, cost::Function, 
	prices::Price, horizon::Int64; order::Int64=1)

	"""compute value functions: return Dict(1=>Array ... horizon=>Array)

	train_noise > noise training data
	controls, states > discretized control and state spaces 
	dynamics > function(x, u, w) returning next state
	cost > function(p, x, u, w) returning stagewise cost 
	price > price per period
	horizon > time horizon
	order > interpolation order

	"""

	state_size = size(states)
	state_steps = grid_steps(states)
	state_iterator = run(states, enumerate=true)
	control_iterator = run(controls)

	value_functions = Dict(t => zeros(state_size...) for t in 1:horizon)

	@showprogress for t in horizon-1:-1:1

		value_function = value_functions[t]

		price = prices[t+1]
		noise_iterator = run(train_noises, t+1)
		interpolator = interpolate(value_functions[t+1], BSpline(Linear()))
		
		for (state, index) in state_iterator

			state = collect(state)
	
			value_w = Float64[]
			proba_w = Float64[]

			for (noise, probability) in noise_iterator

				noise = collect(noise)
				p_noise = prod(probability)
				v = Inf

				for control in control_iterator

					control = collect(control)
					next_state = dynamics(t, state, control, noise)

					if !admissible_state(next_state, states)
						continue
					end

					where = next_state ./ state_steps .+ 1.
					next_value_function = interpolator(where...)

					v = min(v, cost(t, price, state, control, noise) + next_value_function)

				end

				push!(value_w, v)
				push!(proba_w, p_noise)

			end

			expectation = value_w'*proba_w

			value_function[index...] = expectation

		end

		value_functions[t] = value_function

	end

	return value_functions

end

function compute_mean_risk_value_functions(train_noises::Union{Noise, Array{Noise}}, 
	controls::Grid, states::Grid, dynamics::Function, cost::Function, 
	prices::Price, horizon::Int64, lambda::Float64, alpha::Float64,
	; order::Int64=1)

	"""compute value functions: return Dict(1=>Array ... horizon=>Array)

	train_noise > noise training data
	controls, states > discretized control and state spaces 
	dynamics > function(x, u, w) returning next state
	cost > function(p, x, u, w) returning stagewise cost
	price > price per period
	horizon > time horizon
	lambda > weight of risk measure (1-λ)E[z]+λAV@R[z]
    α > P[z<V@Rα] = 1-α
	order > interpolation order

	"""

	state_size = size(states)
	state_steps = grid_steps(states)
	state_iterator = run(states, enumerate=true)
	control_iterator = run(controls)

	value_functions = Dict(t => zeros(state_size...) for t in 1:horizon+1)

	@showprogress for t in horizon-1:-1:1

		value_function = value_functions[t]

		price = prices[t+1]
		noise_iterator = run(train_noises, t+1)
		interpolator = interpolate(value_functions[t+1], BSpline(Linear()))
		
		for (state, index) in state_iterator

			state = collect(state)

			value_w = Float64[]
			proba_w = Float64[]

			for (noise, probability) in noise_iterator

				noise = collect(noise)
				p_noise = prod(probability)
				v = Inf

				for control in control_iterator

					control = collect(control)
					next_state = dynamics(state, control, noise)

					if !admissible_state(next_state, states)
						continue
					end

					where = next_state ./ state_steps .+ 1.
					next_value_function = interpolator(where...)

					v = min(v, cost(price, state, control, noise) + next_value_function)

				end

				push!(value_w, v)
				push!(proba_w, p_noise)

			end

			expectation = value_w'*proba_w
			var_alpha = StatsBase.quantile(value_w, AnalyticWeights(proba_w), 1-alpha)
			avar_alpha = var_alpha + max.(0, value_w .- var_alpha)'*proba_w / alpha

			value_function[index...] = lambda*expectation + (1-lambda)*avar_alpha

		end

		value_functions[t] = value_function

	end

	return value_functions

end

function compute_online_policy(t::Int64, x::Array{Float64}, w::Noise, price::Array{Float64},
	states::Grid, control_iterator, value_function::Array{Float64}, dynamics::Function, 
	cost::Function, state_steps::Array{Float64})

	"""compute online policy: return optimal control at state x observing w

	x > current state
	w > observed noise 
	price > price at current stage
	states > discretized state space
	control_iterator > iterator over control space
	interpolator > value function interpolator
	dynamics > function(x, u, w) returning next state
	cost > function(p, x, u, w) returning stagewise cost
	state_steps > state grid steps

	"""

	interpolator = interpolate(value_function, BSpline(Linear()))
	noise_iterator = run(w, 1)
	vopt = Inf
    uopt = 0.
        
    for control in control_iterator
        
        control = collect(control)
        value_w = Float64[]
		proba_w = Float64[]
		pass = false

        for (noise, probability) in noise_iterator

			noise = collect(noise)
			p_noise = prod(probability)

			next_state = dynamics(t, x, control, noise)

			if !admissible_state(next_state, states)
				pass = true
				break
			end

			where = next_state ./ state_steps .+ 1.
			next_value_function = interpolator(where...)

			v = cost(t, price, x, control, noise) + next_value_function

			push!(value_w, v)
			push!(proba_w, p_noise)

		end

		if pass
			continue
		end

		expectation = value_w'*proba_w

        if expectation < vopt

            vopt = expectation
            uopt = control
        
        end
        
    end

    return uopt

end

function compute_online_trajectory(x0::Union{Float64, Array{Float64}}, test_noise::Array{Float64},
	value_functions::Dict{Int64, T}, controls::Grid, states::Grid, dynamics::Function,
	cost::Function, prices::Price, horizon::Int64; order::Int64=1) where T <: Array{Float64}

	"""compute online trajectory: return 

	x0 > (admissible) init state
	test_noise > online noise scenario 
	value_function > offline computed value functions
	controls, states >  discretized control and state spaces
	dynamics > function(x, u, w) returning next state
	cost > function(p, x, u, w) returning stagewise cost
	price > price per period
	horizon > time horizon
	order > interpolation order

	"""

	dimension = length(x0)
    online_stock = zeros(horizon+1, dimension)
    online_cost = zeros(horizon, 1)
    online_stock[1, :] = x0
    state = x0

    state_steps = grid_steps(states)
    control_iterator = run(controls)
    
    for t in 1:1:horizon
        
        noise = test_noise[t, :]
        price = prices[t]
        value_function = value_functions[t]

        uopt = compute_online_policy(state, noise, price, states, control_iterator, value_function,
        	dynamics, cost, state_steps)
        
        online_cost[t] = cost(price, state, uopt, noise)
        state = dynamics(state, uopt, noise)
        online_stock[t+1, :] = state
        
    end
    
    return online_stock, online_cost
    
end
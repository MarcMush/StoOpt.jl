# developed with Julia 1.1.1
#
# offline step for Stochastic Dynamic Programming 


# SDP 

function compute_expected_realization(sdp::SdpModel, variables::Variables, 
	interpolation::Interpolation)

	realizations = Float64[] 
	probabilities = Float64[]
	reject_control = Float64[]

	for (noise, probability) in iterator(variables.noise)

		noise = collect(noise)
		next_state = sdp.dynamics(variables.t, variables.state, variables.control, noise)

		if !admissible_state!(next_state, sdp.states)
			push!(reject_control, probability)
		end

		next_value_function = eval_interpolation(next_state, interpolation)
		realization = sdp.cost(variables.t, variables.state, variables.control, noise) + 
			next_value_function

		push!(realizations, realization)
		push!(probabilities, probability)

	end

	if isapprox(sum(reject_control), 1.0)
		return Inf
	else
		expected_cost_to_go = realizations'*probabilities
		return expected_cost_to_go
	end

end 

function compute_cost_to_go(sdp::SdpModel, variables::Variables, interpolation::Interpolation)

	cost_to_go = Inf

	for control in sdp.controls.iterator

		variables.control = collect(control)
		realization = compute_expected_realization(sdp, variables, interpolation)
		cost_to_go = min(cost_to_go, realization)

	end

	return cost_to_go

end

# """
# original, not distributed
# """
# function fill_value_function!(sdp::SdpModel, variables::Variables, 
#     value_functions::ArrayValueFunctions, interpolation::Interpolation)

#     value_function = ones(size(sdp.states))

#     for (state, index) in sdp.states.iterator

#         variables.state = collect(state)
#         value_function[index...] = compute_cost_to_go(sdp, variables, interpolation)

#     end

#     value_functions[variables.t] = value_function

#     return nothing
# end

"""
distributed with @distributed and SharedArrays

allocations can probably be reduced
"""
function fill_value_function!(sdp::SdpModel, variables::Variables, 
	value_functions::ArrayValueFunctions, interpolation::Interpolation)

	value_function = SharedArray{Float64}(size(sdp.states))

	@sync @distributed for (state, index) in collect(sdp.states.iterator)

		variables.state = collect(state)
		value_function[index...] = compute_cost_to_go(sdp, variables, interpolation)

	end

	value_functions[variables.t] = value_function

	return nothing
end


# """
# distributed with pmap
# """
# function fill_value_function!(sdp::SdpModel, variables::Variables, 
#     value_functions::ArrayValueFunctions, interpolation::Interpolation)

#     value_functions[variables.t] = pmap(Iterators.product(sdp.states.axis...)) do state
#         variables.state = collect(state)
#         return compute_cost_to_go(sdp, variables, interpolation)
#     end

#     return nothing

# end

function initialize_value_functions(sdp::SdpModel)

	value_functions = ArrayValueFunctions((sdp.horizon+1, size(sdp.states)...))

	if !isnothing(sdp.final_cost)

		final_values = zeros(size(sdp.states))
		for (state, index) in sdp.states.iterator
			state = collect(state)
			final_values[index...] = sdp.final_cost(state)
		end
		value_functions[sdp.horizon+1] = final_values
		
	end

	return value_functions

end

function compute_value_functions(sdp::SdpModel)

	value_functions = initialize_value_functions(sdp)

	for t in sdp.horizon:-1:1

		variables = Variables(t, RandomVariable(sdp.noises, t))
		interpolation = Interpolation(sdp.states, interpolate(value_functions[t+1],
			BSpline(Linear())))

		fill_value_function!(sdp, variables, value_functions, interpolation)

	end

	return value_functions

end

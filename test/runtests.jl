# developed with Julia 1.1.1
#
# tests for StoOpt package

using StoOpt, JLD, Test
using Interpolations

current_directory = @__DIR__


@testset "StoOpt" begin

    @testset "struct" begin 
        
        function test_grid_iterator()
        	g = Grid(1:3.0, 10:12.0, enumerate=true)
        	for (val, index) in g.iterator
        		if index == (2, 3) && val == (2.0, 12.0)
        			return 1
        		end
        	end
        	return nothing
        end

        function test_noise_iterator_1d()
        	w = reshape(collect(1:6.0), 3, 2)
        	pw = ones(3, 2)*0.5
        	noises = Noises(w, pw)
            noise = RandomVariable(noises, 3)
        	for (val, proba) in StoOpt.iterator(noise)
        		if (val[1], proba) == (6.0, 0.5)
        			return 1
        		end
        	end
        	return nothing
        end

        function test_noise_iterator_2d()
        	w = reshape(collect(1:12.0), 3, 2, 2)
        	pw = ones(3, 2)*0.5
        	noises = Noises(w, pw)
            noise = RandomVariable(noises, 3)
        	for (val, proba) in StoOpt.iterator(noise)
        		if (val, proba) == ([6.0, 12.0], 0.5)
        			return 1
        		end
        	end
        	return nothing
        end

        function test_noise_kmeans()
            data = rand(5, 100)
            noise = Noises(data, 3)
            pw = noise.pw.data
            if size(pw)!= (5, 3)
                return nothing
            else
                for i in sum(pw, dims=2)
                    if !isapprox(i, 1.0)
                        return nothing
                    end
                end
            end
            return 1
        end

        @test test_grid_iterator() == 1
        @test test_noise_iterator_1d() == 1
        @test test_noise_iterator_2d() == 1
        @test test_noise_kmeans() == 1

    end

    data = load(current_directory*"/data/test.jld")
    noises = Noises(data["w"], data["pw"])
    states = Grid(0:0.1:1, enumerate=true)
    controls = Grid(-1:0.1:1)
    horizon = 96
    price = ones(96)*0.1
    price[28:84] .+= 0.05
    cmax = 5.
    umax = 1.
    r = 0.95

    function cost(t::Int64, x::Array{Float64,1}, u::Array{Float64,1}, w::Array{Float64,1})
        u = u[1]*umax
        demand = u + w[1]
        return price[t]*max(0.,demand) 
    end

    function dynamics(t::Int64, x::Array{Float64,1}, u::Array{Float64,1}, w::Array{Float64,1})
        scale_factor = umax/cmax
        return x + (r*max.(0.,u) - max.(0.,-u)/r)*scale_factor
    end

    @testset "SDP" begin
            
            sdp = StoOpt.SDP(states, controls, noises, cost, dynamics, horizon)

            interpolation = StoOpt.Interpolation(states, interpolate(zeros(11), BSpline(Linear())))
            variables = StoOpt.Variables(horizon, [0.0], [-0.1], RandomVariable(noises, horizon))

            @test StoOpt.compute_expected_realization(sdp, variables, interpolation) == Inf

            @test isapprox(StoOpt.compute_cost_to_go(sdp, variables, interpolation),
                0.048178640538937376)

            value_functions = StoOpt.ArrayValueFunctions((sdp.horizon, size(sdp.states)...))
            StoOpt.fill_value_function!(sdp, variables, value_functions,
                interpolation)

            @test isapprox(value_functions[horizon][11], 0.0012163218646055842,)

            t = @elapsed value_functions = compute_value_functions(sdp)

            @test t < 8.
            @test value_functions[horizon+1] == zeros(size(states))
            @test all(value_functions[1] .> value_functions[horizon])
            @test value_functions[1][1] > value_functions[1][end]
            @test compute_control(sdp, 1, [0.0], RandomVariable(noises, 1), 
                value_functions) == [0.0]

            sdp.final_cost = f(x::Array{Float64,1}) = 0.02*x[1]
            t = @elapsed value_functions = compute_value_functions(sdp)

            @test t < 8.
            @test value_functions[horizon+1] == 0.02*collect(states.axis...)
    end
    
end


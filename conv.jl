using LinearAlgebra, SparseArrays, Plots
Plots.default(show=true)

# Heat equation : Temporal Convergence
N_values = [20,40,80,160,320,640,1280]
dt_values = [1/(2N + 1) for N in N_values]
l2_values = [0.10359466488537553, 0.06432210014033757, 0.036158765108191306, 0.01903234181347069, 0.009538625691453816,0.004532936542666415,0.0019619319531950586]
l2_values_c = [0.1365838350573606, 0.09103055659783614, 0.05258844899647746, 0.02510223844085985, 0.007579844230312373,0.002101513946762172,0.0022054991693548852]

plot(dt_values, l2_values, label="Implicit", marker=:circle, xscale=:log10, yscale=:log10, title="Temporal Convergence", xlabel="dt", ylabel="L2 Error")
plot!(dt_values, l2_values_c, label="Crank", marker=:circle)
plot!(dt_values, dt_values, label="dt", linestyle=:dash)
plot!(dt_values, dt_values.^2, label="dt^2", linestyle=:dash)
readline()

# Print Order of convergence
ooc = [log10(l2_values[i]/l2_values[i+1])/log10(dt_values[i]/dt_values[i+1]) for i in 1:length(N_values)-1]
ooc_c = [log10(l2_values_c[i]/l2_values_c[i+1])/log10(dt_values[i]/dt_values[i+1]) for i in 1:length(N_values)-1]
println("Implicit : ", ooc)
println("Crank : ", ooc_c)
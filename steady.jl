using LinearAlgebra, SparseArrays, Plots
Plots.default(show=true)
N = 160       # N points intéreurs
a = 0.0      # Borne inférieure
b = 1.0      # Borne supérieure
L = b - a   # Longueur du domaine
ϵ = 0.0001   # Cut-cell parameter
h = 1 / (2N + 1) # Pas de maillage
α₁ = 1.0     # Coefficient Robin \alpha à gauche
β₁ = 0.0     # Coefficient Robin \beta à gauche
α₂ = 1.0     # Coefficient Robin \alpha à droite
β₂ = 0.0     # Coefficient Robin \beta à droite
Rₐ = 0.0    # Valeur CL à gauche
Rᵦ = 0.0    # Valeur CL à droite
T = 1.0     # Temps final
Nt = 1000000   # Nombre d'incréments
dt = T/Nt   # Pas de temps

S(x) = sin(pi*x)
exact(x) = -1/(pi^2)*sin(pi*x) # S(x)=sin(pi*x)

function solve_poisson_cut_cell(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)
    x_f = [a + 2i*h for i in 0:N]
    x_f2 = [x_f[1]-2ϵ*L; x_f; x_f[end]+2ϵ*L]
    x = [(x_f2[i]+x_f2[i+1])/2 for i in 1:N+2]

    A = spzeros(N+2,N+2)
    b = zeros(N+2)
    A[1,1] = α₁ + β₁/ϵ
    A[1,2] = -β₁/ϵ
    b[1] = Rₐ

    A[2,1] = 1/ϵ
    A[2,2] = -(1/ϵ + 1/(h+ϵ))
    A[2,3] = 1/(h+ϵ)
    b[2] = 2*ϵ*L^2*S(x[1])

    A[3,2] = 1/(h+ϵ)
    A[3,3] = -(1/(2h)+1/(ϵ+h))
    A[3,4] = 1/(2h)
    b[3] = 2*h*L^2*S(x[2])

    for i in 4:N-1
        A[i,i-1] = 1/2h
        A[i,i] = -1/h
        A[i,i+1] = 1/2h
        b[i] = 2*h*L^2*S(x[i-1])
    end

    A[N,N] = -(1/(2h)+1/(ϵ+h))
    A[N,N+1] = 1/(h+ϵ)
    A[N,N-1] = 1/(2h)
    b[N] = 2*h*L^2*S(x[N-1])

    A[N+1,N+1] = -(1/ϵ + 1/(h+ϵ))
    A[N+1,N+2] = 1/ϵ
    A[N+1,N] = 1/(h+ϵ)
    b[N+1] = 2*ϵ*L^2*S(x[N])

    A[N+2,N+1] = β₂/ϵ
    A[N+2,N+2] = α₂ - β₂/ϵ
    b[N+2] = Rᵦ

    # Solve
    u = A \ b

    u_ω = u[2:end-1] 
    u_ω_cut = [u[2];u[end-1]]
    u_ω_fluid = u[3:end-2]
    u_γ = [u[1];u[end]]

    return u_ω, u_ω_cut, u_ω_fluid, u_γ, x

end

function solve_poisson_lagrange(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)
    x_f = [a + 2i*h for i in 0:N]
    x_f2 = [x_f[1]-2ϵ*L; x_f; x_f[end]+2ϵ*L]
    x = [(x_f2[i]+x_f2[i+1])/2 for i in 1:N+2]

    f = ϵ
    g = f+h

    A = spzeros(N+2,N+2)
    b = zeros(N+2)
    A[1,1] = α₁ + β₁/ϵ
    A[1,2] = -β₁/ϵ
    b[1] = Rₐ

    A[2,1] = 4/(f+g)
    A[2,2] = (-4*f-g)/(f*g)
    A[2,3] = (4*f)/((f+g)*g)
    b[2] = 2*ϵ*L^2*S(x[1])

    A[3,2] = 1/(h+ϵ)
    A[3,3] = -(1/(2h)+1/(ϵ+h))
    A[3,4] = 1/(2h)
    b[3] = 2*h*L^2*S(x[2])

    for i in 4:N-1
        A[i,i-1] = 1/2h
        A[i,i] = -1/h
        A[i,i+1] = 1/2h
        b[i] = 2*h*L^2*S(x[i-1])
    end

    A[N,N] = -(1/(2h)+1/(ϵ+h))
    A[N,N+1] = 1/(h+ϵ)
    A[N,N-1] = 1/(2h)
    b[N] = 2*h*L^2*S(x[N-1])

    A[N+1,N+1] = (-4*f-g)/(f*g)
    A[N+1,N+2] = 4/(f+g)
    A[N+1,N] = 4*f/((f+g)*g)
    b[N+1] = 2*ϵ*L^2*S(x[N])

    A[N+2,N+1] = β₂/ϵ
    A[N+2,N+2] = α₂ - β₂/ϵ
    b[N+2] = Rᵦ

    # Solve
    u = A \ b

    u_ω = u[2:end-1] 
    u_ω_cut = [u[2];u[end-1]]
    u_ω_fluid = u[3:end-2]
    u_γ = [u[1];u[end]]

    return u_ω, u_ω_cut, u_ω_fluid, u_γ, x

end

u_ω, u_ω_cut, u_ω_fluid, u_γ, x = solve_poisson_cut_cell(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)
u_ω_l, u_ω_cut_l, u_ω_fluid_l, u_γ_l, x_l = solve_poisson_lagrange(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)

exact_ = [exact(x_) for x_ in x[2:end-1]]

# Plot u_ω
plot(x[2:end-1], u_ω, title="Poisson - u_ω", label = "Numerical - Cut Cell")
plot!(x[2:end-1], u_ω_l, label = "Numerical - Lagrange")
plot!(x[2:end-1],exact_, label="Analytical")
readline()

# Plot u_ω_cut
plot([x[1];x[end]], u_ω_cut, title="Poisson - u_ω_cut", label = "u_ω_cut[1]", marker=:circle)
plot!([x[1];x[end]], u_ω_cut_l, label = "u_ω_cut[1] - Lagrange", marker=:circle)
plot!([x[1];x[end]], [exact(x[2]);exact(x[end-1])], label="Analytical")
readline()

# Plot u_ω_fluid
plot(x[3:end-2], u_ω_fluid, title="Poisson - u_ω_fluid", label = "u_ω_fluid[1]", marker=:circle)
plot!(x[3:end-2], u_ω_fluid_l, label = "u_ω_fluid[1] - Lagrange", marker=:circle)
plot!(x[3:end-2],exact_[2:end-1], label="Analytical")
readline()

# Plot u_γ
plot([x[1];x[end]], u_γ, title="Poisson - u_γ", label = "u_γ[1]", marker=:circle)
plot!([x[1];x[end]], u_γ_l, label = "u_γ[1] - Lagrange", marker=:circle)
readline()

# Plot Error u_ω
error = u_ω - exact_
error_l = u_ω_l - exact_
plot(x[2:end-1], error, title="Poisson - u_ω", label = "Error - Cut Cell")
plot!(x[2:end-1], error_l, label = "Error - Lagrange")
readline()

# Plot Error u_ω_cut
error_cut = u_ω_cut - [exact(x[1]);exact(x[end])]
error_cut_l = u_ω_cut_l - [exact(x[1]);exact(x[end])]
plot([x[1];x[end]], error_cut, title="Poisson - u_ω_cut", label = "Error", marker=:circle)
plot!([x[1];x[end]], error_cut_l, label = "Error - Lagrange", marker=:circle)
readline()

# Plot Error u_ω_fluid
error_fluid = u_ω_fluid - exact_[2:end-1]
error_fluid_l = u_ω_fluid_l - exact_[2:end-1]
plot(x[3:end-2], error_fluid, title="Poisson - u_ω_fluid", label = "Error", marker=:circle)
plot!(x[3:end-2], error_fluid_l, label = "Error - Lagrange", marker=:circle)
readline()

# Plot Error u_γ
error_γ = u_γ - [exact(x[1]);exact(x[end])]
error_γ_l = u_γ_l - [exact(x[1]);exact(x[end])]
plot([x[1];x[end]], error_γ, title="Poisson - u_γ", label = "Error", marker=:circle)
plot!([x[1];x[end]], error_γ_l, label = "Error - Lagrange", marker=:circle)
readline()

# Print L1 Norm
function l1_norm(error, ϵ, h, L)
    l1 = (sum(2*h*L*abs.(error[2:end-1])) + 2*ϵ*L*abs(error[1]) + 2*ϵ*L*abs(error[end]))/(2*2*ϵ*L + 2*h*L*(N-2))
    return l1
end
L1_norm = l1_norm(error, ϵ, h, L)
println("L1 Norm All Cells: ", L1_norm)

L1_norm_cut = (2*ϵ*L*abs(error_cut[1]) + 2*ϵ*L*abs(error_cut[end]))/(2*2*ϵ*L)
println("L1 Norm Cut Cells: ", L1_norm_cut)

L1_norm_fluid = 2*h*L*sum(abs.(error_fluid))/(2*h*L*(N-2))
println("L1 Norm Fluid Cells: ", L1_norm_fluid)

# Print Max Norm
println("Max Norm : ", maximum(abs.(error)))

L1_norm_l = l1_norm(error_l, ϵ, h, L)
println("L1 Norm All Cells - Lagrange: ", L1_norm_l)

L1_norm_cut_l = (2*ϵ*L*abs(error_cut_l[1]) + 2*ϵ*L*abs(error_cut_l[end]))/(2*2*ϵ*L)
println("L1 Norm Cut Cells - Lagrange: ", L1_norm_cut_l)

L1_norm_fluid_l = 2*h*L*sum(abs.(error_fluid_l))/(2*h*L*(N-2))
println("L1 Norm Fluid Cells - Lagrange: ", L1_norm_fluid_l)

# Print Max Norm
println("Max Norm - Lagrange: ", maximum(abs.(error_l)))


# Plot Max Norms
plot(title="Max Norms", xlabel="ϵ", ylabel="Max Norm", xscale=:log10, yscale=:log10, legend=:bottomright)
for i in 1:length(N_values)
    plot!(ϵ_values, Max_norms[:,i], label="N = $(N_values[i])", marker=:circle)
end
plot!(ϵ_values, ϵ_values, label="O(ϵ)", linestyle=:dash)
plot!(ϵ_values, ϵ_values.^2, label="O(ϵ^2)", linestyle=:dash)
readline()

# Plot L1 Norms
plot(title="L1 Norms", xlabel="ϵ", ylabel="L1 Norm", xscale=:log10, yscale=:log10, legend=:bottomright)
for i in 1:length(N_values)
    plot!(ϵ_values, L1_norms[:,i], label="N = $(N_values[i])", marker=:circle)
end
plot!(ϵ_values, ϵ_values, label="O(ϵ)", linestyle=:dash)
plot!(ϵ_values, ϵ_values.^2, label="O(ϵ^2)", linestyle=:dash)
readline()

"""

# ϵ Value Sensitivity on L1, Max Norms
N_values = [20,40,80,160,320,640,1280]
ϵ_values = 0.0001:0.001:1.0
L1_norms = zeros(length(ϵ_values),length(N_values))
Max_norms = zeros(length(ϵ_values),length(N_values))

for i in 1:length(ϵ_values)
    for j in 1:length(N_values)
        println("ϵ = $(ϵ_values[i]), N = $(N_values[j])")
        N = N_values[j]
        h = 1 / (2N + 1)
        u_ω, u_ω_cut, u_ω_fluid, u_γ, x = solve_poisson_cut_cell(N,a,b,L,ϵ_values[i],h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)
        exact_ = [exact(x_) for x_ in x[2:end-1]]
        error = u_ω - exact_
        L1_norms[i,j] = l1_norm(error, ϵ_values[i], h, L)
        Max_norms[i,j] = maximum(abs.(error))
    end
end
"""
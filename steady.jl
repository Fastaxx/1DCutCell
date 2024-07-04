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
    x = [a + 2i*h for i in 0:N+1]


    A = spzeros(N+2,N+2)
    b = zeros(N+2)
    A[1,1] = α₁ + β₁/ϵ
    A[1,2] = -β₁/ϵ
    b[1] = Rₐ

    A[2,1] = 1/ϵ
    A[2,2] = -(1/ϵ + 1/(h+ϵ))
    A[2,3] = 1/(h+ϵ)
    b[2] = 2*ϵ*L^2*S(x[2])

    A[3,2] = 1/(h+ϵ)
    A[3,3] = -(1/(2h)+1/(ϵ+h))
    A[3,4] = 1/(2h)
    b[3] = 2*h*L^2*S(x[3])

    for i in 4:N-1
        A[i,i-1] = 1/2h
        A[i,i] = -1/h
        A[i,i+1] = 1/2h
        b[i] = 2*h*L^2*S(x[i])
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
    u_γ = [u[1];u[end]]
    return u_ω, u_γ, x

end


u_ω, u_γ, x = solve_poisson_cut_cell(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)

exact_ = [exact(x_) for x_ in x[2:end-1]]

# Plot
plot(x[2:end-1], u_ω, title="Poisson - u_ω", label = "Numerical")
plot!(x[2:end-1],exact_, label="Analytical")
readline()

# Plot
plot([x[1];x[end]], u_γ, title="Poisson - u_γ", label = "u_γ[1]", marker=:circle)
readline()

# Plot Error
error= u_ω - exact_
plot(x[2:end-1], error)
readline()

# Print Max Norm
println("Max Norm", maximum(abs.(error)))

# Norm : 
# N : 1280                   640                     320                     160
# E-max : 0.0003091374432173191  0.0006815419916040562   0.0014251382010778545   0.0029072828242329026


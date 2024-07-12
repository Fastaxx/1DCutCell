using LinearAlgebra, SparseArrays, Plots
Plots.default(show=true)
N = 160       # N points intéreurs
a = 0.0      # Borne inférieure
b = 1.0      # Borne supérieure
L = b - a   # Longueur du domaine
ϵ = 0.0001   # Cut-cell parameter
h = L / (N-1) # Pas de maillage
α₁ = 1.0     # Coefficient Robin \alpha à gauche
β₁ = 0.0     # Coefficient Robin \beta à gauche
α₂ = 0.0     # Coefficient Robin \alpha à droite
β₂ = 0.0     # Coefficient Robin \beta à droite
Rₐ = 0.0    # Valeur CL à gauche
Rᵦ = 0.0    # Valeur CL à droite

S(x) = sin(pi*x)
exact(x) = -1/(pi^2)*sin(pi*x) # S(x)=sin(pi*x)
exact_p(x) = -1/pi*cos(pi*x)

# Compact Scheme Regular Grid : Poisson Equation

function solve_poisson_1d_compact_scheme(a, b, N, hx, S, Rₐ, Rᵦ)
    x = [a + i*hx for i in 0:N-1]

    A = zeros(2N-1, 2N-1)
    b_vector = zeros(2N-1)

    # Build the Matrix A
    for i in 1:N-2
        A[i, i] = -1.0
        A[i, i+1] = 1.0
    end

    A[N-1,N-1] = 1.0
    A[N-1,N-2] = 0
    A[N-1,N-3] = 0
    A[N-1,end] = -1/hx
    A[N-1,end-1] = 1/hx
    A[N-1,end-2] = 0
    A[N-1,end-3] = 0

    A[N,N] = 1.0 # T0 = Rₐ
    A[2N-1,2N-1] = 1.0 # TN = Rᵦ

    A[N+1,1] = 1.0
    A[N+1,2] = 0
    A[N+1,3] = 0
    A[N+1,N] = 1/hx
    A[N+1,N+1] = -1/hx
    A[N+1,N+2] = 0
    A[N+1,N+3] = 0

    for i in N+2:2N-2
        A[i, i-1] = 6/(5*hx)
        A[i, i] = -6/(5*hx)
        A[i, i-N-1] = 1/10
        A[i, i-N] = 1
        A[i, i-N+1] = 1/10
    end

    # Build the right-hand side
    for i in 1:N-2
        b_vector[i] = hx*S(x[i])
    end

    b_vector[N-1] = 0
    b_vector[N] = Rₐ
    b_vector[2N-1] = Rᵦ

    b_vector[N+1] = 0.0
    for i in N+2:2N-2
        b_vector[i] = 0.0
    end

    # Solve the system
    u = A \ b_vector

    q = u[1:N-1]
    u_solution = u[N:2N-1]

    return q, u_solution, x
end

q, u, x = solve_poisson_1d_compact_scheme(a, b, N, h, S, Rₐ, Rᵦ)

# Plot the solution
scatter(x, u, label="Solution", xlabel="x", ylabel="u(x)", legend=:bottomright, marker=:circle)
scatter!(x, [exact(x[i]) for i in 1:N], label="Exact solution", linestyle=:dash)
readline()

# Plot the Flux
plot(q, label="Flux", xlabel="x", ylabel="q(x)", legend=:bottomright)
plot!([exact_p((x[i] + x[i+1])/2) for i in 1:N-1], label="Exact flux", linestyle=:dash)
readline()

# Plot the error
error = u - [exact(x[i]) for i in 1:N]
plot(x, error, label="Error", xlabel="x", ylabel="Error", legend=:bottomright)
readline()

# Print the error
error = u - [exact(x[i]) for i in 1:N]

l1 = (sum(abs.(error[2:end-1]))*h)/(N-2)
println("L1 Norm : ", l1)

println("Max Norm : ", maximum(abs.(error)))

# Convergence study
N_values = [20,40,80,160,320,640,1280]


for N in N_values
    h = 1 / (N + 1)
    q, u, x = solve_poisson_1d_compact_scheme(a, b, N, h, S, Rₐ, Rᵦ)
    error = u - [exact(x[i]) for i in 1:N]
    l1 = (sum(abs.(error[2:end-1]))*h)/(N-2)
    println("N = $N, L1 Norm : ", l1)
    println("N = $N, Max Norm : ", maximum(abs.(error)))
end

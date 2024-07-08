using LinearAlgebra, SparseArrays, Plots
Plots.default(show=true)
N = 1280       # N points intéreurs
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
Nt = 100000   # Nombre d'incréments
dt = T/Nt   # Pas de temps

function solve_heat_cut_cell_implicit(N,a,b,L,ϵ,h,Rₐ,Rᵦ,dt,T,u,f,g,S)
    x_f = [a + 2i*h for i in 0:N]
    x_f2 = [x_f[1]-2ϵ*L; x_f; x_f[end]+2ϵ*L]
    x = [(x_f2[i]+x_f2[i+1])/2 for i in 1:N+2]

    A = spzeros(N+2,N+2)
    b = zeros(N+2)

    A[1,1] = 1.0

    A[2,1] = -1/(ϵ*L)
    A[2,2] = 1/(ϵ*L) + 1/(h*L+ϵ*L) + 2h*L/dt
    A[2,3] = -1/(h*L+ϵ*L)

    A[3,2] = -1/(h*L+ϵ*L)
    A[3,3] = (1/(h*L+ϵ*L) + 1/(2*h*L) + 2*h*L/dt)
    A[3,4] = -1/(2*h*L)

    for i in 4:N-1
        A[i,i-1] = -1/(2h*L)
        A[i,i] = (1/(h*L) + 2*h*L/dt)
        A[i,i+1] = -1/(2h*L)
    end

    A[N,N] = (1/(h*L+ϵ*L) + 1/(2*h*L) + 2*h*L/dt)
    A[N,N-1] = -1/(2*h*L)
    A[N,N+1] = -1/(h*L+ϵ*L)

    A[N+1, N+1] = (1/(ϵ*L) + 1/(h*L+ϵ*L) + 2h*L/dt)
    A[N+1, N+2] = -1/(ϵ*L)
    A[N+1, N] = -1/(h*L+ϵ*L)


    A[N+2,N+2]=1.0

    t=0.0
    while t<T
        b[1] = f(t+dt)
        b[2] = 2*ϵ*L*S(x[2],t+dt) + u[1]*2*ϵ*L/dt
        b[3] = 2*h*L*S(x[3],t+dt) + u[2]*2*h*L/dt
        for i in 4:N-1
            b[i] = 2*h*L*S(x[i],t+dt) + u[i]*2*h*L/dt
        end
        b[N] = 2*h*L*S(x[N-1],t+dt) + u[N]*2*h*L/dt
        b[N+1] = 2*ϵ*L*S(x[N],t+dt) + u[N+1]*2*ϵ*L/dt
        b[N+2] = g(t+dt)

        # Solve 
        u_new = A \ b

        # Update
        u = u_new
        t+=dt
    end

    u_ω = u[2:end-1] 
    u_γ = [u[1];u[end]]

    return u_ω, u_γ, x
end

function solve_heat_cut_cell_crank(N,a,b,L,ϵ,h,Rₐ,Rᵦ,dt,T,u,f,g,S)
    x_f = [a + 2i*h for i in 0:N]
    x_f2 = [x_f[1]-2ϵ*L; x_f; x_f[end]+2ϵ*L]
    x = [(x_f2[i]+x_f2[i+1])/2 for i in 1:N+2]

    A = spzeros(N+2,N+2)
    b = zeros(N+2)

    A[1,1] = 1.0

    A[2,1] = -1/(ϵ*L)*0.5
    A[2,2] = 1/(ϵ*L)*0.5 + 1/(h*L+ϵ*L)*0.5 + 2h*L/dt
    A[2,3] = -1/(h*L+ϵ*L)*0.5

    A[3,2] = -1/(h*L+ϵ*L)*0.5
    A[3,3] = (1/(h*L+ϵ*L)*0.5 + 1/(2*h*L)*0.5 + 2*h*L/dt)
    A[3,4] = -1/(2*h*L)*0.5

    for i in 4:N-1
        A[i,i-1] = -1/(2h*L)*0.5
        A[i,i] = (1/(h*L)*0.5 + 2*h*L/dt)
        A[i,i+1] = -1/(2h*L)*0.5
    end

    A[N,N] = (1/(h*L+ϵ*L)*0.5 + 1/(2*h*L)*0.5 + 2*h*L/dt)
    A[N,N-1] = -1/(2*h*L)*0.5
    A[N,N+1] = -1/(h*L+ϵ*L)*0.5

    A[N+1, N+1] = (1/(ϵ*L)*0.5 + 1/(h*L+ϵ*L)*0.5 + 2h*L/dt)
    A[N+1, N+2] = -1/(ϵ*L)*0.5
    A[N+1, N] = -1/(h*L+ϵ*L)*0.5


    A[N+2,N+2]=1.0

    t=0.0
    while t<T
        b[1] = 0.5*(f(t) + f(t+dt))
        b[2] = 0.5*(2*ϵ*L*S(x[2],t) + 2*ϵ*L*S(x[2],t+dt)) + u[1]*(2*ϵ*L/dt - 0.5*1/(h*L+ϵ*L) - 0.5*1/(ϵ*L)) + u[2]*0.5*1/(h*L+ϵ*L) + u[1]*0.5*1/(ϵ*L)
        b[3] = 0.5*(2*h*L*S(x[3],t) + 2*h*L*S(x[3],t+dt)) + u[2]*(2*h*L/dt - 0.5*1/(2*h*L) - 0.5*1/(h*L+ϵ*L)) + u[3]*0.5*1/(2*h*L) + u[1]*0.5*1/(h*L+ϵ*L)
        for i in 4:N-1
            b[i] = 0.5*(2*h*L*S(x[i],t)+2*h*L*S(x[i],t+dt)) + u[i]*(2*h*L/dt-0.5*2/(2*h*L)) + u[i-1]*0.5*1/(2*h*L) + u[i+1]*0.5*1/(2*h*L)
        end
        b[N] = 0.5*(2*h*L*S(x[N-1],t) + 2*h*L*S(x[N-1],t+dt)) + u[N]*(2*h*L/dt - 0.5*1/(2*h*L) - 0.5*1/(h*L+ϵ*L)) + u[N-1]*0.5*1/(2*h*L) + u[N+1]*0.5*1/(h*L+ϵ*L)
        b[N+1] = 0.5*(2*ϵ*L*S(x[N],t) + 2*ϵ*L*S(x[N],t+dt)) + u[N+1]*(2*ϵ*L/dt - 0.5*1/(ϵ*L) - 0.5*1/(h*L+ϵ*L)) + u[N]*0.5*1/(h*L+ϵ*L) + u[N+2]*0.5*1/(ϵ*L)
        b[N+2] = 0.5*(g(t) + g(t+dt))

        # Solve 
        u_new = A \ b

        # Update
        u = u_new
        t+=dt
    end

    u_ω = u[2:end-1] 
    u_γ = [u[1];u[end]]

    return u_ω, u_γ, x
end

x_f = [a + 2i*h for i in 0:N]
x_f2 = [x_f[1]-2ϵ*L; x_f; x_f[end]+2ϵ*L]
x = [(x_f2[i]+x_f2[i+1])/2 for i in 1:N+2]

# Exact Solution
exact_heat(x,t) = exp(-t)*cos(x)
exact_heat_ = [exact_heat(x_,T) for x_ in x[2:end-1]]

# Boundary values
f(t) = exp(-t)
g(t) = exp(-t) * cos(1)

# Source Value
S(x,t) = 0.0

# Initialisation
u = [cos(x_) for x_ in x]

# Solve
u_ω, u_γ, x = solve_heat_cut_cell_implicit(N,a,b,L,ϵ,h,Rₐ,Rᵦ,dt,T,u,f,g,S)
u_ω_c, u_γ_c, x_c = solve_heat_cut_cell_crank(N,a,b,L,ϵ,h,Rₐ,Rᵦ,dt,T,u,f,g,S)

# Plot
plot(x[2:end-1], u_ω, title="Heat - u_ω", label = "Implicit")
plot!(x[2:end-1],u_ω_c, label="Crank")
plot!(x[2:end-1],exact_heat_, label="Analytical")
readline()

# Plot
plot([x[1];x[end]], u_γ, title="Heat - u_γ", label = "u_γ - Numerical - Implicit", marker=:circle)
plot!([x[1];x[end]], u_γ_c, label = "u_γ - Numerical - Crank", marker=:circle)
plot!([x[1];x[end]], [f(T);g(T)], label = "u_γ - Analytical")
readline()


function compute_norm(error, ϵ, h, L)
    l2 = sqrt(sum((error[2:end-1]).^2*(2*h*L)) + error[1]^2*(2*ϵ*L) + error[end]^2*(2*ϵ*L)/(2*2*ϵ*L + 2*h*L*(N-2)))
    return l2
end

# Plot Error
error = u_ω - exact_heat_
error_c = u_ω_c - exact_heat_
plot(x[2:end-1], error, label="Implicit")
plot!(x[2:end-1], error_c, label="Crank")
readline()

# Print
println("L2 Norm")
l2 = compute_norm(error, ϵ, h, L)
l2_c = compute_norm(error_c, ϵ, h, L)
println("L2 Norm - Implicit : ", l2)
println("L2 Norm - Crank : ",l2_c)

println("Max Norm")
println("Max Norm - Implicit : ", maximum(abs.(error)))
println("Max Norm - Crank : ", maximum(abs.(error_c)))


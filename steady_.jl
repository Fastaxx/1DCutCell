using LinearAlgebra, SparseArrays, Plots
Plots.default(show=true)
N = 20       # N points intéreurs
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

S(x) = sin(pi*x)
exact(x) = -1/(pi^2)*sin(pi*x) # S(x)=sin(pi*x)
exact_p(x) = -1/pi*cos(pi*x)

# Compact Scheme : Poisson Equation

function flux_matrix(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)
    x_f = [a + 2i*h for i in 0:N] # Faces régulières    
    x_f2 = [x_f[1]-2ϵ*L; x_f; x_f[end]+2ϵ*L] # Ajout de 2 cut cells
    x = [(x_f2[i]+x_f2[i+1])/2 for i in 1:N+2] # Centres des cellules
    xf = [(x[1]+x_f2[1])/2; (x[2]+x[1])/2; [(x[i]+x[i+1])/2 for i in 2:N-1]; (x[end-1]+x[end])/2; (x[end]+x_f2[end])/2] # Localisation des flux

    # Flux Matrix
    A_q = spzeros(N+1,N+1)
    b_q = zeros(N+1)

    A_q[1,1] = 1.0
    A_q[1,2] = -1.0
    b_q[1] = 2*ϵ*L*S(x[1])

    for i in 2:N-1
        A_q[i,i+1] = 1.0
        A_q[i,i] = -1.0
        b_q[i] = 2*h*L*S(x[i])
    end

    A_q[N,N] = -1.0
    A_q[N,N+1] = 1.0
    b_q[N] = 2*ϵ*L*S(x[N-1])

    A_q[N+1,N+1] = 1.0
    b_q[N+1] = exact_p(xf[end])

    u = A_q\b_q

    return u, x, xf
end

function gradient_matrix(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)
    u, x, xf = flux_matrix(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)

    # Gradient Matrix
    A = spzeros(N+2,N+2)
    b = zeros(N+2)

    A[1,1] = -1/(ϵ*L)
    A[1,2] = 1/(ϵ*L)
    b[1] = u[1]

    A[2,2] = -1/(h*L+ϵ*L)
    A[2,3] = 1/(h*L+ϵ*L)
    b[2] = u[2]

    for i in 3:N-1
        A[i,i+1] = 1/(2*h*L)
        A[i,i] = -1/(2*h*L)
        b[i] = u[i]
    end

    A[N,N+1] = 1/(h*L+ϵ*L)
    A[N,N] = -1/(h*L+ϵ*L)
    b[N] = u[N]

    A[N+1,N+1] = -1/(ϵ*L)
    A[N+1,N+2] = 1/(ϵ*L)
    b[N+1] = u[N+1]

    A[N+2,N+2] = 1.0
    b[N+2] = exact(x[end])

    T = A\b

    return T, x, xf
end


function gradient_matrix_compact(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)
    u, x, xf = flux_matrix(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)

    f=ϵ*L
    g = f + h*L
    hx = h


    q1 = [1; 0; 0] 
    q3 = [g^2/(3*(f+g)*(f+2*g+hx)); (3*f*(g+hx) + g*(2*g+3*hx))/(3*(f+g)*(g+hx)); g^2/(3*(g+hx)*(f+2*g+hx))]
    q5 = [hx^2/(3*(g^2+4*g*hx+3*hx^2)); (6*g+5*hx)/(6*(g+hx)); hx/(6*(g+3*hx))]
    centered = [5/60; 5/6; 5/60]


    #q1 = [3/4, 1/4, 0]
    #q3 = [(6*f^2-6*f*g+6*f*hx+2*g^2-3*g*hx)/(24*f^2+12*f*hx); (6*f^2+6*f*hx-2*g^2+3*g*hx)/(12*f*hx); (g^2-3*f^2)/(3*hx*(2*f*hx))]
    #q5 = [(6*f^2-12*f*g+12*f*hx+6*g^2-12*g*hx+5*hx^2)/(12*hx^2); (-3*f^2+6*f*g-3*f*hx-3*g^2+3*g*hx+2*hx^2)/(3*hx^2); (6*f^2-12*f*g+6*g^2-hx^2)/(12*hx^2)]

    # Gradient Matrix
    A = spzeros(N+2,N+2)
    b = zeros(N+2)

    A[1,1] = -1/(ϵ*L)
    A[1,2] = 1/(ϵ*L)
    b[1] = u[1]*q1[1]

    A[2,2] = -1/(h*L+ϵ*L)
    A[2,3] = 1/(h*L+ϵ*L)
    b[2] = u[2]*q3[2] + u[1]*q3[1] + u[3]*q3[3]

    A[3,3] = -1/(2*h*L)
    A[3,4] = 1/(2*h*L)
    b[3] = q5[1]*u[2] + q5[2]*u[3] + q5[3]*u[4]

    for i in 4:N-2
        A[i,i+1] = 1/(2*h*L)
        A[i,i] = -1/(2*h*L)
        b[i] = centered[1]*u[i-1]+centered[2]*u[i]+centered[3]*u[i+1]
    end

    A[N-1,N-1] = -1/(2*h*L)
    A[N-1,N] = 1/(2*h*L)
    b[N-1] = q5[1]*u[N-2] + q5[2]*u[N-1] + q5[3]*u[N]

    A[N,N+1] = 1/(h*L+ϵ*L)
    A[N,N] = -1/(h*L+ϵ*L)
    b[N] = u[N]*q3[2] + u[N-1]*q3[3] + u[N-2]*q3[1]

    A[N+1,N+1] = -1/(ϵ*L)
    A[N+1,N+2] = 1/(ϵ*L)
    b[N+1] = u[N+1]*q1[1]

    A[N+2,N+2] = 1.0
    b[N+2] = exact(x[end])

    T = A\b

    return T, x, xf
end

function compute_norm(error, ϵ, h, L)
    l2 = sqrt(sum((error[2:end-1]).^2*(2*h*L)) + error[1]^2*(2*ϵ*L) + error[end]^2*(2*ϵ*L)/(2*2*ϵ*L + 2*h*L*(N-2)))
    return l2
end

u, x, xf = flux_matrix(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)

exact_p_ = [exact_p(xf[i]) for i in 1:N+1]

plot(u, label="Numerical Solution - Flux", marker=:circle)
plot!(exact_p_, label="Exact Solution - Flux", marker=:circle)
readline()



T, x, xf = gradient_matrix(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)
T_c, x_c, xf_c = gradient_matrix_compact(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)

exact_ = [exact(x[i]) for i in 1:N]


plot(T, label="Numerical Solution - Gradient", marker=:circle)
plot!(T_c, label="Numerical Solution - Compact", marker=:circle)
plot!(exact_, label="Exact Solution - Gradient", marker=:circle)
readline()

plot(T[2:end-1]-exact_, label="Error - Gradient", marker=:circle)
plot!(T_c[2:end-1]-exact_, label="Error - Compact", marker=:circle)
readline()


# Print Max Values
println("Max Error : ", maximum(abs.(T[2:end-1]-exact_)))
println("Max Error - Compact: ", maximum(abs.(T_c[2:end-1]-exact_)))

# Print L2 Norm
println("L2 Norm : ", compute_norm(T[2:end-1]-exact_, ϵ, h, L))
println("L2 Norm - Compact: ", compute_norm(T_c[2:end-1]-exact_, ϵ, h, L))


"""
plot(u, label="Numerical Solution - Flux", marker=:circle)
plot!(exact_p_, label="Exact Solution - Flux", marker=:circle)
readline()

plot(u-exact_p_, label="Error - Flux", marker=:circle)
readline()


# Print Max Values
println("Max Error : ", maximum(abs.(u-exact_p_)))
l1 = 2*ϵ*L*abs(u[1]-exact_p(xf[1])) + 2*ϵ*L*abs(u[end]-exact_p(xf[end])) + sum(abs.(u[2:end-1]-exact_p_[2:end-1])*2*h*L)
println("L1 Error: ", l1)

# Convergence study
N_values=[20,40,80,160,320,640,1280]
for N in N_values
    h = 1 / (2N + 1)
    u, x, xf = flux_matrix(N,a,b,L,ϵ,h,α₁,β₁,α₂,β₂,Rₐ,Rᵦ,S)
    exact_p_ = [exact_p(xf[i]) for i in 1:N+1]
    max_error = maximum(abs.(u-exact_p_))
    l1 = 2*ϵ*L*abs(u[1]-exact_p(xf[1])) + 2*ϵ*L*abs(u[end]-exact_p(xf[end])) + sum(abs.(u[2:end-1]-exact_p_[2:end-1])*2*h*L)
    println("N = ", N, " Max Error : ", max_error)
end
"""
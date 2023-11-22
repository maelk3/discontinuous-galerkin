"""
        numerical simulation of inviscid 1d burgers equation using a
        nodal discontinuous gakerkin spectral element method with a
        TVDM flux limiter and an explicit 3DRSSPRK timestepping
"""

using Plots
using LinearAlgebra
using FastGaussQuadrature

N::Int64 = 100 # number of cells
n::Int64 = 4   # polynomial degree

h::Float64 = 1/N

ξ::Vector{Float64}, ω::Vector{Float64} = gausslobatto(n+1) # quadrature nodes and weights

M::Matrix{Float64}     = h/2*diagm(ω)      # mass matrix
M_inv::Matrix{Float64} = 2/h*diagm(1 ./ ω) # inverse mass matrix

x::Matrix{Float64} = repeat(range(0, 1-h, step=h), 1, n+1) + repeat(h/2*(ones(n+1)+ξ), 1, N)' # global grid points

u_0::Matrix{Float64} = zeros(N, n+1)
u::Matrix{Float64} = zeros(N, n+1)
v::Matrix{Float64} = zeros(N, n+1)

u_1::Matrix{Float64} = zeros(N, n+1)
v_1::Matrix{Float64} = zeros(N, n+1)

u_2::Matrix{Float64} = zeros(N, n+1)
v_2::Matrix{Float64} = zeros(N, n+1)

RHS::Matrix{Float64} = zeros(N, n+1)
flux::Matrix{Float64} = zeros(N, n+1)

c::Float64 = 1.0 # max |f'(u)| maximum wave speed
eta::Float64 = 1.0
dt::Float64 = eta*h/c*1/(2*n+1)        # timestep
T::Float64 = 0.17                      # final time
nb_iterations::Int64 = Int(ceil(T/dt)) # number of iterations

D::Matrix{Float64} = zeros(n+1, n+1)   # differentiation matrix
for p in 1:n+1, q in 1:n+1
    if p == q
        D[p,q] = sum(1 ./ (ξ[p]*ones(n+1) .- ξ)[begin:p-1]) + sum(1 ./ (ξ[p]*ones(n+1) .- ξ)[p+1:end])
    else
        D[p,q] = 1
        for m in 1:n+1
            if m != p
                D[p,q] /= ξ[p]-ξ[m]
                if m != q
                    D[p,q] *= ξ[q]-ξ[m]
                end
            end
        end
    end
end

S::Matrix{Float64} = 2/h*D*M # stiffness matrix

scalar_product::Matrix{Float64} = [(1+(-1)^(i+j-2))/(i+j-1) for i in 1:n+1, j in 1:n+1] # scalar product matrix for the basis {1, X, …, X^n}

_, R::Matrix{Float64} = qr(sqrt(scalar_product))

normalized_legendre_basis = inv(R)

V::Matrix{Float64} = [ξ[i] ^ (j-1) for i in 1:n+1, j in 1:n+1]

legendre_basis::Matrix{Float64} = normalized_legendre_basis*diagm(reshape((V[end,:]'*normalized_legendre_basis) .^ (-1), n+1))

nodal_to_modal::Matrix{Float64} = inv(V*legendre_basis) # change of basis matrix from Lagrange interpolation polynomial to Legendre polynomials
modal_to_nodal::Matrix{Float64} = inv(nodal_to_modal)

minmod(a::Float64, b::Float64, c::Float64) = sign(a) == sign(b) && sign(b) == sign(c) ? sign(a)*min(abs(a), abs(b), abs(c)) : 0.0

modes::Matrix{Float64} = zeros(N, 2)
mode::Vector{Float64} = zeros(n+1)

"""
        Flux limiter function. This function compute the flux limited
        variable v from the variable u
"""
function flux_limiter!(u::Matrix{Float64}, v::Matrix{Float64}) # minmod flux limiter
    for k in 1:N
        mul!(view(modes, k, :), view(nodal_to_modal, 1:2,:), view(u, k, :))
    end
    mean  = @view modes[:,1]
    slope = @view modes[:,2]

   for i in 1:N
        if minmod(2*slope[i]/h, (mean[mod(i, N)+1]-mean[i])/(h/2), (mean[i]-mean[mod(i-2, N)+1])/(h/2)) != 2*slope[i]/h
            mode[1] = mean[i]
            mode[2] = h/2*minmod(2*slope[i]/h, (mean[mod(i, N)+1]-mean[i])/(h/2), (mean[i]-mean[mod(i-2, N)+1])/(h/2))
            mul!(view(v, i, :), modal_to_nodal, mode)
        else
            v[i,:] .= @view u[i,:]
        end
    end

    return nothing
end

"""
        Discontinuous Galerkin discretisation operator of the scalar
        conservation law, This function computes inplace of RHS the
        discretization of the conservation law.
"""
function ∇_DG!(u::Matrix{Float64}, RHS::Matrix{Float64})
    flux .= f.(u)
    for k in 1:N
        mul!(view(RHS, k, :), S, view(flux, k, :))
        RHS[k,end]   -= f_hat(u[k,end], u[mod(k, N)+1,begin])
        RHS[k,begin] += f_hat(u[mod(k-2,N)+1,end], u[k,begin])
    end

    return nothing
end

"""
        3rd order TVD Runge Kutta time discretization using the minmod
        flux limiter
"""
function SSP_runge_kutta!(nb_iterations::Int64)
    for t in 1:nb_iterations
        ∇_DG!(v, RHS)
        mul!(u_1, RHS, M_inv)
        u_1 .*= dt
        u_1 .+= v
        flux_limiter!(u_1, v_1)

        ∇_DG!(v_1, RHS)
        mul!(u_2, RHS, M_inv)
        u_2 .*= 1/4*dt
        u_2 .+= 3/4 .* v .+ 1/4 .* v_1
        flux_limiter!(u_2, v_2)

        ∇_DG!(v_2, RHS)
        mul!(u, RHS, M_inv)
        u .*= 2/3*dt
        u .+= 1/3 .* v .+ 2/3 .* v_2
        flux_limiter!(u, v)
    end

    return nothing
end

f(u::Float64) = u^2/2                         # flux
f_hat(u::Float64, v::Float64) = (f(u)+f(v))/2 # numerical flux

# initialization
u_0 .= (x -> sin(2π*x)).(x)
u   .= u_0
flux_limiter!(u, v)

@time SSP_runge_kutta!(nb_iterations)

# final time plot
p = plot(reshape(x', (n+1)*N), reshape(u_0', (n+1)*N)) # initial condition
plot!(reshape(x', (n+1)*N), reshape(v', (n+1)*N), marker=(:cross, 2)) # final time discretization
plot!(reshape(x', (n+1)*N), reshape(repeat(mapslices((x -> nodal_to_modal[1:1,:]*x), v, dims=2), 1, n+1)', (n+1)*N)) # mean values
plot!(reshape(x', (n+1)*N), reshape(mapslices(x -> modal_to_nodal*x, cat(mapslices(x -> nodal_to_modal[1:2,:]*x, v, dims=2)', zeros(N, n-1)', dims=1)', dims=2)', (n+1)*N)) # flux limited 2 order reconstruction

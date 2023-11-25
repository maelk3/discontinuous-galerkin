"""
        numerical simulation of euler equations using a nodal
        discontinuous gakerkin spectral element method with a TVDM
        flux limiter and an explicit 3DRSSPRK timestepping with HLLE
        flux. Sod shock tube test cast
"""

using Plots
using LinearAlgebra
using FastGaussQuadrature

N::Int64 = 200  # number of cells
n::Int64 = 4    # polynomial degree

h::Float64 = 1/N

ξ::Vector{Float64}, ω::Vector{Float64} = gausslobatto(n+1) # quadrature nodes and weights

M::Matrix{Float64}     = h/2*diagm(ω)      # mass matrix
M_inv::Matrix{Float64} = 2/h*diagm(1 ./ ω) # inverse mass matrix

x::Matrix{Float64} = repeat(range(0, 1-h, step=h), 1, n+1) + repeat(h/2*(ones(n+1)+ξ), 1, N)' # global grid points

q_0::Array{Float64, 3} = zeros(N, n+1, 3)
q::Array{Float64, 3} = zeros(N, n+1, 3)
q_lim::Array{Float64, 3} = zeros(N, n+1, 3)

aux::Array{Float64, 3} = zeros(N, n+1, 4)

q_1::Array{Float64, 3} = zeros(N, n+1, 3)
q_lim_1::Array{Float64, 3} = zeros(N, n+1, 3)

q_2::Array{Float64, 3} = zeros(N, n+1, 3)
q_lim_2::Array{Float64, 3} = zeros(N, n+1, 3)

RHS::Array{Float64, 3} = zeros(N, n+1, 3)

flux::Array{Float64, 3} = zeros(N, n+1, 3)

c::Float64 = 1.0 # max |f'(q)| maximum wave speed
eta::Float64 = 1.0
T::Float64 = 0.2 # final time

D::Matrix{Float64} = zeros(n+1, n+1) # differentiation matrix
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

modes::Array{Float64, 3} = zeros(N, 2, 3)
mode::Vector{Float64} = zeros(n+1)

"""
        Flux limiter function. This function compute the flux limited
        variable q_lim from the variable q
"""
function flux_limiter!(q::Array{Float64, 3}, q_lim::Array{Float64, 3}) # minmod flux limiter
    for k in 1:N, i in 1:3
        mul!(view(modes, k, :, i), view(nodal_to_modal, 1:2,:), view(q, k, :, i))
    end
    mean  = @view modes[:,1,:]
    slope = @view modes[:,2,:]

    for k in 1:N, i in 1:3
        if minmod(2*slope[k,i]/h, (mean[mod(k, N)+1,i]-mean[k,i])/(h/2), (mean[k,i]-mean[mod(k-2, N)+1,i])/(h/2)) != 2*slope[k,i]/h
            mode[1] = mean[k,i]
            mode[2] = h/2*minmod(2*slope[k,i]/h, (mean[mod(k, N)+1,i]-mean[k,i])/(h/2), (mean[k,i]-mean[mod(k-2, N)+1,i])/(h/2))
            mul!(view(q_lim, k, :, i), modal_to_nodal, mode)
        else
            q_lim[k,:, i] .= @view q[k,:, i]
        end
    end
    return nothing
end

function compute_auxiliary_variables!(q::Array{Float64, 3}, aux::Array{Float64, 3})
    ρ  = @view q[:,:,1]
    ρu = @view q[:,:,2]
    ρe = @view q[:,:,3]

    u = @view aux[:,:,1]
    p = @view aux[:,:,2]
    H = @view aux[:,:,3]
    c = @view aux[:,:,4]

    @. u = ρu / ρ
    @. p = (γ-1)*(ρe-1/2*ρ*u^2)
    @. H = (ρe+p)/ρ
    @. c = sqrt((γ-1)*(H-1/2*u^2))

    return nothing
end

û::Vector{Float64} = zeros(N-1)
Ĥ::Vector{Float64} = zeros(N-1)
ĉ::Vector{Float64} = zeros(N-1)
function compute_roe_average_auxiliary_variables!(q_l, q_r, aux_l, aux_r, û, Ĥ, ĉ)
    ρ_l = @view q_l[:,1]
    ρ_r = @view q_r[:,1]

    u_l = @view aux_l[:,1]
    p_l = @view aux_l[:,2]
    H_l = @view aux_l[:,3]
    c_l = @view aux_l[:,4]

    u_r = @view aux_r[:,1]
    p_r = @view aux_r[:,2]
    H_r = @view aux_r[:,3]
    c_r = @view aux_r[:,4]

    @. û = (sqrt(ρ_l)*u_l+sqrt(ρ_r)*u_r)/(sqrt(ρ_l)+sqrt(ρ_r))
    @. Ĥ = (sqrt(ρ_l)*H_l+sqrt(ρ_r)*H_r)/(sqrt(ρ_l)+sqrt(ρ_r))
    @. ĉ = sqrt((γ-1)*(Ĥ-1/2*û^2))

    return nothing
end

S_l::Vector{Float64} = zeros(N-1)
S_r::Vector{Float64} = zeros(N-1)
function compute_speed_averages!(aux_l, aux_r, û, ĉ, S_l, S_r)
    u_l = @view aux_l[:,1]
    c_l = @view aux_l[:,4]

    u_r = @view aux_r[:,1]
    c_r = @view aux_r[:,4]

    @. S_l = min(û-ĉ, u_l-c_l)
    @. S_r = max(û+ĉ, u_r+c_r)

    return nothing
end

function compute_flux!(q::Array{Float64, 3},
                       aux::Array{Float64, 3},
                       flux::Array{Float64, 3})
    ρ  = @view q[:,:,1]
    ρu = @view q[:,:,2]
    ρe = @view q[:,:,3]

    p = @view aux[:,:,2]

    F_ρ  = @view flux[:,:,1]
    F_ρu = @view flux[:,:,2]
    F_ρe = @view flux[:,:,3]

    @. F_ρ  = ρu
    @. F_ρu = ρu^2/ρ+p
    @. F_ρe = ρu/ρ*(ρe+p)

    return nothing
end

f_HLLE::Matrix{Float64} = zeros(N-1, 3)
S_rp::Matrix{Float64} = zeros(N-1, 3)
S_lm::Matrix{Float64} = zeros(N-1, 3)
function compute_intercell_hlle_fluxes!(f_HLLE, S_rp, S_lm, f_l, f_r, q_l, q_r)
    @views @. f_HLLE = (S_rp*f_l-S_lm*f_r)/(S_rp-S_lm) + S_rp*S_lm/(S_rp-S_lm)*(q_r-q_l)

    return nothing
end

"""
        Discontinuous Galerkin discretisation operator of the scalar
        conservation law, This function computes inplace of RHS the
        discretization of the conservation law.
"""
function ∇_DG!(q::Array{Float64, 3}, RHS::Array{Float64, 3})
    q_l = @view q[1:end-1,end,:]
    q_r = @view q[2:end,begin,:]

    aux_l = @view aux[1:end-1,end,:]
    aux_r = @view aux[2:end,begin,:]

    f_l = @view flux[1:end-1,end,:]
    f_r = @view flux[2:end,begin,:]

    compute_auxiliary_variables!(q, aux)
    compute_flux!(q, aux, flux)

    compute_roe_average_auxiliary_variables!(q_l, q_r, aux_l, aux_r, û, Ĥ, ĉ)
    compute_speed_averages!(aux_l, aux_r, û, ĉ, S_l, S_r)

    for k in 1:N-1, i in 1:3
        S_lm[k,i] = min(S_l[k], 0.0)
        S_rp[k,i] = max(S_r[k], 0.0)
    end

    compute_intercell_hlle_fluxes!(f_HLLE, S_rp, S_lm, f_l, f_r, q_l, q_r)

    for k in 1:N, i in 1:3
        mul!(view(RHS, k, :, i), S, view(flux, k, :, i))
        if k < N
            RHS[k,end, i] -= f_HLLE[k,i]
        else
            RHS[k,end, i] -= f_HLLE[end,i]
        end
        if k > 1
            RHS[k,begin,i] += f_HLLE[k-1,i]
        else
            RHS[k,begin,i] += f_HLLE[1,i]
        end
    end

    return nothing
end

"""
        3rd order TVD Runge Kutta time discretization using the minmod
        flux limiter
"""
function SSP_runge_kutta!()
    t = 0
    c_max = 3.0
    dt  = eta*h/c_max*1/(2*n+1)
    while t < T
        ∇_DG!(q_lim, RHS)
        for k in 1:N, i in 1:3
            mul!(view(q_1, k, :, i), M_inv, view(RHS, k, :, i))
        end
        q_1 .*= dt
        q_1 .+= q_lim
        flux_limiter!(q_1, q_lim_1)

        ∇_DG!(q_lim_1, RHS)
        for k in 1:N, i in 1:3
            mul!(view(q_2, k, :, i), M_inv, view(RHS, k, :, i))
        end

        q_2 .*= 1/4*dt
        q_2 .+= 3/4 .* q_lim .+ 1/4 .* q_lim_1
        flux_limiter!(q_2, q_lim_2)

        ∇_DG!(q_lim_2, RHS)
        for k in 1:N, i in 1:3
            mul!(view(q, k, :, i), M_inv, view(RHS, k, :, i))
        end

        q .*= 2/3*dt
        q .+= 1/3 .* q_lim .+ 2/3 .* q_lim_2
        flux_limiter!(q, q_lim)

        t += dt
        c_max = max(-minimum(S_l), maximum(S_r))
        dt  = eta*h/c_max*1/(2*n+1)
        println("t:", t, ", dt:", dt)
    end

    return nothing
end

# initialization
γ::Float64 = 1.4

# sod shock tube initial conditions
ρ₀(x) = x < 1/2 ? 1.0 : 0.125
ρu₀(x) = 0.0
p₀(x) = x < 1/2 ? 1.0 : 0.1
ρe₀(x) = p₀(x)/(γ-1)

q_0[:,:,1] = ρ₀.(x)
q_0[:,:,2] = ρu₀.(x)
q_0[:,:,3] = ρe₀.(x)

q   .= q_0
flux_limiter!(q, q_lim)

@time SSP_runge_kutta!()

# final time plot
p = plot(reshape(x', (n+1)*N), reshape(q_0[:,:,1]', (n+1)*N)) # initial condition
plot!(reshape(x', (n+1)*N), reshape(q[:,:,1]', (n+1)*N), marker=(:cross, 2)) # final time discretization
plot!(reshape(x', (n+1)*N), reshape(repeat(mapslices((x -> nodal_to_modal[1:1,:]*x), q[:,:,1], dims=2), 1, n+1)', (n+1)*N)) # mean values
plot!(reshape(x', (n+1)*N), reshape(mapslices(x -> modal_to_nodal*x, cat(mapslices(x -> nodal_to_modal[1:2,:]*x, q[:,:,1], dims=2)', zeros(N, n-1)', dims=1)', dims=2)', (n+1)*N)) # flux limited 2 order reconstruction

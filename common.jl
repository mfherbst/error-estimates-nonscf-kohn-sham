import Pkg
Pkg.activate(".")

import DFTK: index_G_vectors, eval_psp_projection_radial
import SpecialFunctions: erf
using ProgressMeter
using DFTK
using DoubleFloats
using GenericLinearAlgebra
using HDF5
using IntervalArithmetic
using LinearAlgebra
using Polynomials
using Printf
using Roots
using TimerOutputs

import FFTW
FFTW.set_num_threads(1)

const to = TimerOutput()

###
### Helper functions
###

@doc raw"""
Returns the result of the integral ``∫_τ^∞ t^k exp(-t^2 / σ) dt``
"""
function integral_monomial_gaussian(k::Integer, τ::T; σ=1, upper_bound_ok=false) where T
    @assert k ≥ -4
    @assert σ > 0
    τdσ = τ / T(sqrt(σ))  # τ divided by σ

    ## Even k
    if k == -4
        exp(-τdσ^2) / 3τ^3 - 2 / T(3σ) * integral_monomial_gaussian(-2, τ, σ=σ)
    elseif k == -2
        sqrt(T(π) / σ)        * (erf(τdσ) - 1) + exp(-τdσ^2) / τ
    elseif k == 0
        sqrt(T(π) * σ)    / 2 * (1 - erf(τdσ))
    elseif k == 2
        sqrt(T(π) * σ^3)  / 4 * (1 - erf(τdσ)) + (                             σ * τ / 2) * exp(-τdσ^2)
    elseif k == 4
        3sqrt(T(π) * σ^5) / 8 * (1 - erf(τdσ)) + (             σ * τ^3 / 2 +  3σ^2*τ / 4) * exp(-τdσ^2)
    elseif k == 6
        15sqrt(T(π) * σ^7)/16 * (1 - erf(τdσ)) + (σ*τ^5 / 2 + 5σ^2*τ^3 / 4 + 15σ^3*τ / 8) * exp(-τdσ^2)
    #
    ## Odd k
    elseif k == -3
        upper_bound_ok || error("The case k == -3 is not implemented.")
        # ∫_τ^∞ (1/t^3) * exp(-t^2 / σ) dt ≤ (1/τ^3) ∫_τ^∞ exp(-t^2 / σ) dt
        integral_monomial_gaussian(0, τ, σ=σ) / τ^3
    elseif k == -1
        upper_bound_ok || error("The case k == -1 is not implemented.")
        # ∫_τ^∞ (1/t) * exp(-t^2 / σ) dt ≤ (1/τ) ∫_τ^∞ exp(-t^2 / σ) dt
        integral_monomial_gaussian(0, τ, σ=σ) / τ
        # Better bound:
        # \intx^\infty t^{-1} e^{-t^2} dt is
        # = \int{x^2} s^{-1} e^{-s} ds (use the change of variable s=t^2) is e^{-x^2}/(2x^2)
    elseif k == 1
        (                                 σ) / 2 * exp(-τdσ^2)
    elseif k == 3
        (                   σ * τ^2 +   σ^2) / 2 * exp(-τdσ^2)
    elseif k == 5
        (         σ * τ^4 + 2σ^2*τ^2 + 2σ^3) / 2 * exp(-τdσ^2)
    elseif k == 7
        (σ*τ^6 + 3σ^2*τ^4 + 6σ^3*τ^2 + 6σ^4) / 2 * exp(-τdσ^2)
    #
    ## Fallback by recursion
    else
        σ * τ^(k - 1) / 2 * exp(-τdσ^2) + σ * (k - 1) / 2 * integral_monomial_gaussian(k - 2, τ, σ=σ)
    end
end

@doc raw"""
Returns the result of the integral ``∫_τ^∞ t^k P(t) exp(-t^2 / σ) dt``
where `P` is a polynomial in `t`
"""
function integral_polynomial_gaussian(P::Poly, τ::T; k=0, σ=1, upper_bound_ok=false) where T
    # -1 translates from index in coeff to power in t
    sum(coeff * integral_monomial_gaussian(i + k - 1, τ, σ=σ, upper_bound_ok=upper_bound_ok)
        for (i, coeff) in enumerate(P.a) if coeff != 0)
end

@doc raw"""
Computes an upper bound for the sum ``∑_{|G| > G0} |β G|^k Q(|β G|) exp(-|β G|^2 / σ)``
where all `G` are on a lattice defined by `recip_lattice`, `β` and `σ` are a positive
constants, `k` is integer and `Q` is a polynomial both chosen such that
``t^k Q(t) exp(-t / σ)`` is a decreasing function for `t > G0`.
"""
function bound_sum_polynomial_gaussian(polynomial::Poly{T}, recip_lattice, G0::T;
                                       β=1, σ=1, k=0, Gmin=G0) where {T}
    # Determine dimensionality: Note: Clashes with usual DFTK convention
    @assert !iszero(recip_lattice[:, end])
    m = size(recip_lattice, 1)
    @assert size(recip_lattice) == (m, m)

    # Diameter of recip_lattice unit cell.
    diameter = norm(recip_lattice * ones(m))
    @assert diameter ≥ 0  # We assume lattice to be positively orianted
    @assert Gmin > 0
    @assert G0 ≤ Gmin

    # The terms (depending on β, n and G) we sum over.
    term(β, n, G) = (β * G)^n * polynomial(β * G) * exp(-β*G^2)

    if m == 1
        a = abs(recip_lattice[1, 1])
        start = floor(G0 / a) * a
        @assert start ≤ Gmin
        intm = integral_polynomial_gaussian(polynomial, β * Gmin, k=k, σ=σ,
                                            upper_bound_ok=true) / T(β)
        return  2 / a * ((Gmin - start) * term(β, k, Gmin) + intm)
    elseif m == 2 || m == 3
        prefactor = (m == 2 ? 2T(π) : 4T(π)) / abs(det(recip_lattice))
        start = G0 - diameter

        intm = integral_polynomial_gaussian(polynomial, β * Gmin, σ=σ,
                                            k=k + m - 1, upper_bound_ok=true) / T(β)^m
        result = prefactor * ((Gmin - start) * term(β, k + m - 1, Gmin) + intm)

        for (j, aj) in enumerate(eachcol(recip_lattice))
            newlattice = []
            nRj = 1
            for (k, ak) in enumerate(eachcol(recip_lattice))
                k == j && continue
                angle = dot(aj, ak) / dot(aj, aj)

                nRj = nRj * (2 + ceil(abs(angle)))
                newlatticevector = Vector(ak - angle * aj)
                deleteat!(newlatticevector, j)
                push!(newlattice, newlatticevector)
            end
            newlattice = hcat(newlattice...)

            result += nRj * bound_sum_polynomial_gaussian(polynomial, newlattice,
                                                          G0 - diameter, σ=σ, β=β, k=k, Gmin=Gmin)
        end

        return result
    end
    error("Dimensionality > 3")
end

###
### Determination of minimal Ecut2
###

function is_cohen_bergstresser(model)
    (
           all(at isa ElementCohenBergstresser for (at, pos) in model.atoms)
        && typeof.(model.term_types) == [Kinetic, AtomicLocal]
    )
end


function is_linear_atomic(model)
    (
           all(at isa ElementPsp for (at, pos) in model.atoms)
        && typeof.(model.term_types) == [Kinetic, AtomicLocal, AtomicNonlocal,
                                         Ewald, PspCorrection]
    )
end


"""
Estimate a lower bound for the Ecut2, the energy cutoff on the bigger residual grid,
from the given PSP, reciprocal lattice and Ecut
"""
function psp_minimal_Ecut2(psp::PspHgh, recip_lattice, Ecut)
    T = eltype(recip_lattice)
    minimal_Ecuts = T[]

    # Compute the diameter of the first BZ
    diameter = norm(recip_lattice * Vec3(1, 1, 1))
    @assert diameter ≥ 0

    # Local part: We want sqrt(2 * Ecut2) - norm(G) ≥ qcut, where G in Ecut basis
    qcut = DFTK.qcut_psp_local(T, psp)
    push!(minimal_Ecuts, (qcut + sqrt(2Ecut))^2 / 2)

    # Nonlocal projectors: We want sqrt(2 * Ecut2) - norm(k) ≥ qcut,
    #                      where G outside of Ecut2 basis
    proj_idcs = unique((i, l) for (i, l, _) in DFTK.projector_indices(psp))
    for (i, l) in proj_idcs
        qcut = DFTK.qcut_psp_projection_radial(T, psp, i, l)
        push!(minimal_Ecuts, (qcut + diameter)^2 / 2)
    end

    maximum(minimal_Ecuts)
end


function minimal_Ecut2(basis)
    model = basis.model

    if is_cohen_bergstresser(model)
        element = first(model.atoms[1])
        Gmax = sqrt(maximum(keys(element.V_sym))) * (2π / element.lattice_constant)
        return basis.Ecut + sqrt(2basis.Ecut) * Gmax + Gmax^2 / 2
    elseif is_linear_atomic(model)
        maximum(psp_minimal_Ecut2(atom.psp, model.recip_lattice, basis.Ecut)
                for (atom, positions) in model.atoms)
    else
        error("Not implemented")
    end
end


###
### Bounds for \|P_{B_2}^\perp V P_{B_1} \|  (potential_perpb2_b1)
###

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp Vloc P_{B_1} \|_{op}``

hamblock:  Hamiltonian k-block in small Ecut basis
Ecut2:     Large ecut used for ``B_2``
"""
function bound_potential_perpb2_b1_Vloc(hamblock, Ecut2)
    # Code duplication with bound_perpb2_residual_Vloc

    basis = hamblock.basis
    kpoint = hamblock.kpoint
    @assert any(t isa AtomicLocal for t in basis.model.term_types)
    T = eltype(basis)
    recip_lattice = basis.model.recip_lattice

    accu = zero(T)
    for (element, positions) in basis.model.atoms
        element isa ElementCohenBergstresser && continue
        @assert element isa ElementPsp
        @assert element.psp isa PspHgh
        psp = element.psp

        # We need a bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) / t^4
        # where t = psp.rloc * |G|.
        Qloc = DFTK.psp_local_polynomial(T, psp)
        R(q) = bound_sum_polynomial_gaussian(Qloc * Qloc, recip_lattice, q, β=psp.rloc, k=-4)

        for (i, G) in enumerate(G_vectors(kpoint))
            qcut = sqrt(2Ecut2) - norm(recip_lattice * G)
            accu += R(qcut) * length(positions) / basis.model.unit_cell_volume^2
        end
    end
    sqrt(accu)
end

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp Vnlock P_{B_1} \|_{op}``

hamblock:  Hamiltonian k-block in small Ecut basis
Ecut2:     Large ecut used for ``B_2``
"""
function bound_potential_perpb2_b1_Vnlock(hamblock, Ecut2)
    # Highly related to bound_potential_perpb1_Vnloc, a lot of code duplication
    basis = hamblock.basis
    kpoint = hamblock.kpoint
    model = basis.model
    kpoint = hamblock.kpoint
    T = eltype(basis)
    recip_lattice = model.recip_lattice
    @assert any(t isa AtomicLocal for t in model.term_types)

    # Tabulated values for the L∞ norm of the spherical harmonics Ylm
    Ylm∞ = (sqrt(1 / 4T(π)), sqrt(3 / 4T(π)))  # My guess: (5 / 4π, 7 / 4π)

    # Norms of the G vectors in the Ecut basis (B1)
    Gnorms = [norm(model.recip_lattice * (G + kpoint.coordinate))
              for G in G_vectors(kpoint)]

    function projector_bound(psp, l)
        @assert psp isa PspHgh
        n_proj_l = size(psp.h[l + 1], 1)

        sum_proj_B1 = zeros(T, n_proj_l)
        sum_proj_B2perp = zeros(T, n_proj_l)

        for i in 1:n_proj_l
            # Bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) where t = psp.rp[l+1] * |G|.
            Qnloc = DFTK.psp_projection_radial_polynomial(T, psp, i, l)
            qcut = sqrt(2Ecut2) - norm(recip_lattice * kpoint.coordinate)
            Nli = bound_sum_polynomial_gaussian(Qnloc * Qnloc, recip_lattice,
                                                qcut, β=psp.rp[l + 1])

            psp_radials = eval_psp_projection_radial.(psp, i, l, Gnorms)
            sum_proj_B1[i] = norm(psp_radials) * Ylm∞[l + 1] / sqrt(model.unit_cell_volume)
            sum_proj_B2perp[i] = sqrt(Nli) * Ylm∞[l + 1] / sqrt(model.unit_cell_volume)
        end

        ret = sum_proj_B2perp' * abs.(psp.h[l + 1]) * sum_proj_B1
        @assert ndims(ret) == 0
        ret
    end

    @assert all(element isa ElementPsp for (element, _) in model.atoms)
    sum(maximum(projector_bound.(element.psp, 0:element.psp.lmax)) * length(positions)
        for (element, positions) in model.atoms)
end

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp H P_{B_1} \|_{op}``

hamblock:  Hamiltonian k-block in small Ecut basis
Ecut2:     Large ecut used for ``B_2``
"""
function bound_ham_perpb2_b1(hamblock, Ecut2)
    T = eltype(hamblock.basis)
    model = hamblock.basis.model
    if is_cohen_bergstresser(model)
        @assert Ecut2 ≥ minimal_Ecut2(hamblock.basis)
        return (AtomicLocal=zero(T), )
    elseif is_linear_atomic(model)
        (AtomicLocal=bound_potential_perpb2_b1_Vloc(hamblock, Ecut2),
         AtomicNonlocal=bound_potential_perpb2_b1_Vnlock(hamblock, Ecut2))
    else
        error("Not implemented")
    end
end

###
### Bounds for \|P_{B_2}^\perp V ψnk\|  (perpb2_residual)
###

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp V_loc ψnk\|`` using varying
Sobolev exponents `s`.

hamblock:  Hamiltonian k-block in small Ecut basis
ψnk:       Eigenvector for a particular band and k-point on small ecut
Ecut2:     Large ecut used for ``B_2``
s:         Sobolev exponent
"""
function bound_perpb2_residual_Vloc_sobolev(hamblock, ψnk::AbstractVector, Ecut2; s=2)
    basis = hamblock.basis
    kpoint = hamblock.kpoint
    @assert length(ψnk) == length(G_vectors(kpoint))
    @assert any(t isa AtomicLocal for t in basis.model.term_types)
    T = eltype(basis)
    recip_lattice = basis.model.recip_lattice

    accu = zero(T)
    for (element, positions) in basis.model.atoms
        element isa ElementCohenBergstresser && continue
        @assert element isa ElementPsp
        @assert element.psp isa PspHgh
        psp = element.psp

        # We need a bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) / t^4
        # where t = psp.rloc * |G|.
        Qloc = DFTK.psp_local_polynomial(T, psp)
        R(q) = bound_sum_polynomial_gaussian(Qloc * Qloc, recip_lattice, q, β=psp.rloc, k=-4)

        for (i, G) in enumerate(G_vectors(kpoint))
            Gnorm = norm(recip_lattice * G)
            qcut = sqrt(2Ecut2) - Gnorm
            accu += (1 + Gnorm^2)^s * abs2(ψnk[i]) * R(qcut) * length(positions)
        end
    end

    prefac = sum((1 + norm(recip_lattice * G)^2)^(-s) for G in G_vectors(kpoint))
    sqrt(prefac * accu / basis.model.unit_cell_volume^2)
end



@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp V_loc ψnk\|``

hamblock:  Hamiltonian k-block in small Ecut basis
ψnk:     Eigenvector for a particular band and k-point on small ecut
Ecut2:   Large ecut used for ``B_2``
"""
function bound_perpb2_residual_Vloc(hamblock, ψnk::AbstractVector, Ecut2; α=1)
    basis = hamblock.basis
    kpoint = hamblock.kpoint
    @assert length(ψnk) == length(G_vectors(kpoint))
    @assert any(t isa AtomicLocal for t in basis.model.term_types)
    T = eltype(basis)
    recip_lattice = basis.model.recip_lattice

    accu = zero(T)
    for (element, positions) in basis.model.atoms
        element isa ElementCohenBergstresser && continue
        @assert element isa ElementPsp
        @assert element.psp isa PspHgh
        psp = element.psp

        # We need a bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) / t^4
        # where t = psp.rloc * |G|.
        Qloc = DFTK.psp_local_polynomial(T, psp)
        R(q) = bound_sum_polynomial_gaussian(Qloc * Qloc, recip_lattice, q, β=psp.rloc, k=-4)

        for (i, G) in enumerate(G_vectors(kpoint))
            factor = α == 0 ? one(T) : abs(ψnk[i])^2α
            qcut = sqrt(2Ecut2) - norm(recip_lattice * G)
            accu += factor * R(qcut) * length(positions)
        end
    end

    if α == 0
        prefac = one(T)
    elseif α == 1
        prefac = length(ψnk)
    else
        prefac = sum(@. abs(ψnk)^(2-2α))
    end
    sqrt(prefac * accu) / basis.model.unit_cell_volume
end

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp V_nlock ψnk\|``

hamblock:  Hamiltonian k-block in small Ecut basis
ψnk:       Eigenvector for a particular band and k-point on small Ecut
Ecut2:     Large ecut used for ``B_2``
"""
function bound_perpb2_residual_Vnlock(hamblock, ψnk::AbstractVector, Ecut2)
    basis = hamblock.basis
    kpoint = hamblock.kpoint
    T = eltype(basis)
    recip_lattice = basis.model.recip_lattice
    @assert length(ψnk) == length(G_vectors(kpoint))

    # Compute norm of vector f
    idx_nonlocal = only(findall(t -> t isa AtomicNonlocal, basis.model.term_types))
    Vlock = hamblock.operators[idx_nonlocal]
    fnorm = norm(Vlock.D * (Vlock.P' * ψnk))

    accu = zero(T)
    for (element, positions) in basis.model.atoms
        @assert element isa ElementPsp
        @assert element.psp isa PspHgh
        psp = element.psp

        # Ignore m ... sum over it done implicitly by the (2l+1)
        proj_idcs = unique((i, l) for (i, l, _) in DFTK.projector_indices(psp))
        for (i, l) in proj_idcs
            # Bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) where t = psp.rp[l+1] * |G|.
            Qnloc = DFTK.psp_projection_radial_polynomial(T, psp, i, l)
            qcut = sqrt(2Ecut2) - norm(recip_lattice * kpoint.coordinate)
            N = bound_sum_polynomial_gaussian(Qnloc * Qnloc, recip_lattice,
                                              qcut, β=psp.rp[l + 1])

            accu += (2l + 1) * length(positions) * N
        end
    end

    fnorm * sqrt(accu / 4T(π) / basis.model.unit_cell_volume)
end


@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp H ψnk\|`` if ``ψnk`` is given
on the `Ecut` basis.

hamblock:  Hamiltonian k-block in small `Ecut` basis
ψnk:       Eigenvector for a particular band and k-point on small `Ecut`
Ecut2:     Large ecut used for ``B_2``
"""
function bound_perpb2_residual(hamblock, ψnk::AbstractVector, Ecut2)
    T = eltype(hamblock.basis)
    model = hamblock.basis.model
    if is_cohen_bergstresser(model)
        @assert Ecut2 ≥ minimal_Ecut2(hamblock.basis)
        return (Kinetic=zero(T), AtomicLocal=zero(T))
    elseif is_linear_atomic(model)
        (Kinetic=zero(T),
         AtomicLocal=bound_perpb2_residual_Vloc(hamblock, ψnk, Ecut2),
         AtomicNonlocal=bound_perpb2_residual_Vnlock(hamblock, ψnk, Ecut2),
         Ewald=zero(T),
         PspCorrection=zero(T)
        )
    else
        error("Not implemented")
    end
end


###
### Apply P_{B_1^⟂ ∩ B_2} V ψnk
###

@doc raw"""
Get P_{B_1^⟂ ∩ B_2} op P_{B_1} ψk_Ecut2 as a dense matrix, where ψk_Ecut2 is a matrix
with columns in the Ecut basis.
"""
function apply_op_perpb1capb2(opblock, opblock_Ecut2, ψk_Ecut2::AbstractMatrix)
    basis2 = opblock_Ecut2.basis
    kpoint2 = opblock_Ecut2.kpoint
    kpoint = opblock.kpoint
    T = eltype(basis2)
    @assert kpoint2.coordinate == kpoint.coordinate
    @assert size(ψk_Ecut2, 1) == length(G_vectors(kpoint2))

    b1_in_b2 = indexin(G_vectors(kpoint), G_vectors(kpoint2))
    not_in_b1 = [i for i in 1:length(G_vectors(kpoint2)) if !(i in b1_in_b2)]
    @assert length(b1_in_b2) + length(not_in_b1) == length(G_vectors(kpoint2))

    # Compute the part inside B2 explicitly and select
    # only the indices inside B_1^⟂ ∩ B_2.
    (opblock_Ecut2 * ψk_Ecut2)[not_in_b1, :]
end

@doc raw"""
Get P_{B_1^⟂ ∩ B_2} H P_{B_1} ψk_Ecut2 as a dense matrix, where ψk_Ecut2 is a matrix
with columns in the Ecut2 basis.
"""
function apply_hamiltonian_perpb1capb2(hamblock, hamblock_Ecut2, ψk_Ecut2::AbstractMatrix)
    @assert hamblock_Ecut2.basis.Ecut > hamblock.basis.Ecut
    @assert hamblock_Ecut2.kpoint.coordinate == hamblock.kpoint.coordinate
    @assert hamblock.basis.model == hamblock_Ecut2.basis.model
    basis2 = hamblock_Ecut2.basis

    idx_local = only(findall(t -> t isa AtomicLocal, basis2.model.term_types))
    Vloc = hamblock.operators[idx_local]
    Vloc_Ecut2 = hamblock_Ecut2.operators[idx_local]

    if is_cohen_bergstresser(basis2.model)
        @assert basis2.Ecut ≥ minimal_Ecut2(hamblock.basis)
        apply_op_perpb1capb2(Vloc, Vloc_Ecut2, ψk_Ecut2)
    elseif is_linear_atomic(basis2.model)
        idx_nonlocal = only(findall(t -> t isa AtomicNonlocal,
                                    basis2.model.term_types))
        Vnlock = hamblock.operators[idx_nonlocal]
        Vnlock_Ecut2 = hamblock_Ecut2.operators[idx_nonlocal]

        (apply_op_perpb1capb2(Vloc, Vloc_Ecut2, ψk_Ecut2)
         + apply_op_perpb1capb2(Vnlock, Vnlock_Ecut2, ψk_Ecut2))
    else
        error("Not implemented")
    end
end

###
### Bounds for \|P_{B_2} V ψnk\|
###

@doc raw"""
Computes ``\|P_{B_1^⟂ ∩ B_2} op ψnk\|`` for an operator
"""
function bound_perpb1capb2_residual_operator(opblock, opblock_Ecut2, ψnk)
    basis2 = opblock_Ecut2.basis
    kpoint2 = opblock_Ecut2.kpoint
    kpoint = opblock.kpoint
    T = eltype(basis2)
    @assert kpoint2.coordinate == kpoint.coordinate
    @assert length(ψnk) == length(G_vectors(kpoint))

    error("This is currently broken, because the interface " *
          "of apply_op_perpb1capb2 updated")

    # Apply operator to psi and project onto B_1^⟂ ∩ B_2.
    opψnk = apply_op_perpb1capb2(opblock, opblock_Ecut2, reshape(Array(ψnk), :, 1))

    norm(opψnk)  # Compute L2-norm
end


@doc raw"""
Estimates an upper bound for ``\|P_{B_1}^\perp V_loc ψnk\|``

hamblock:        Hamiltonian k-block in small Ecut basis
hamblock_Ecut2:  Hamiltonian k-block in large Ecut basis
ψnk:             Eigenvector for a particular band and k-point on small Ecut
ψnk_Ecut2:       Eigenvector for a particular band and k-point on large Ecut
"""
function bound_perpb1_residual_Vloc(hamblock, hamblock_Ecut2, ψnk)
    basis2 = hamblock_Ecut2.basis
    idx_local = only(findall(t -> t isa AtomicLocal, basis2.model.term_types))
    Vloc = hamblock.operators[idx_local]
    Vloc_Ecut2 = hamblock_Ecut2.operators[idx_local]

    (   bound_perpb1capb2_residual_operator(Vloc, Vloc_Ecut2, ψnk)
      + bound_perpb2_residual_Vloc(hamblock, ψnk, basis2.Ecut))
end


@doc raw"""
Estimates an upper bound for ``\|P_{B_1}^\perp V_nlock ψnk\|``

hamblock:        Hamiltonian k-block in small Ecut basis
hamblock_Ecut2:  Hamiltonian k-block in large Ecut basis
ψnk:             Eigenvector for a particular band and k-point on small Ecut
ψnk_Ecut2:       Eigenvector for a particular band and k-point on large Ecut
"""
function bound_perpb1_residual_Vnlock(hamblock, hamblock_Ecut2, ψnk)
    basis2 = hamblock_Ecut2.basis
    idx_nonlocal = only(findall(t -> t isa AtomicNonlocal, basis2.model.term_types))
    Vnlock = hamblock.operators[idx_nonlocal]
    Vnlock_Ecut2 = hamblock_Ecut2.operators[idx_nonlocal]

    (   bound_perpb1capb2_residual_operator(Vnlock, Vnlock_Ecut2, ψnk)
      + bound_perpb2_residual_Vnlock(hamblock, ψnk, basis2.Ecut))
 end


@doc raw"""
Estimates an upper bound for ``\|P_{B_1}^\perp V ψnk\|`` if ``ψnk`` is given
on the `Ecut` basis.

hamblock:        Hamiltonian k-block in small Ecut basis
hamblock_Ecut2:  Hamiltonian k-block in large Ecut basis
ψnk:             Eigenvector for a particular band and k-point on small Ecut
ψnk_Ecut2:       Eigenvector for a particular band and k-point on large Ecut
"""
function bound_perpb1_residual(hamblock, hamblock_Ecut2, ψnk)
    @assert hamblock_Ecut2.basis.Ecut > hamblock.basis.Ecut
    @assert hamblock_Ecut2.kpoint.coordinate == hamblock.kpoint.coordinate
    @assert hamblock.basis.model == hamblock_Ecut2.basis.model
    basis2 = hamblock_Ecut2.basis

    if is_cohen_bergstresser(basis2.model)
        @assert basis2.Ecut ≥ minimal_Ecut2(hamblock.basis)
        (AtomicLocal=bound_perpb1_residual_Vloc(hamblock, hamblock_Ecut2, ψnk), )
    elseif is_linear_atomic(basis2.model)
        (AtomicLocal=bound_perpb1_residual_Vloc(hamblock, hamblock_Ecut2, ψnk),
         AtomicNonlocal=bound_perpb1_residual_Vnlock(hamblock, hamblock_Ecut2, ψnk))
    else
        error("Not implemented")
    end
end


###
### Bounds for \|V P_{B_1}^⟂\|_op (potential_perpb1)
###

function bound_potential_perpb1_Vloc(hamblock, hamblock_Ecut2)
    basis2 = hamblock_Ecut2.basis
    kpoint2 = hamblock_Ecut2.kpoint
    model = basis2.model
    Ecut2 = basis2.Ecut
    kpoint = hamblock.kpoint
    T = eltype(basis2)
    @assert kpoint.coordinate == kpoint2.coordinate

    accu = zero(T)
    for (element, positions) in model.atoms
        if element isa ElementPsp
            @assert element.psp isa PspHgh

            # We need a bound for Σ_{G, |G| > qcut} |Q(t)| exp(-t^2 / 2) / t^2
            # where t = psp.rloc * |G|. Since |Q| is a decaying function not going
            # through zero after |G| > qcut, we may replace |Q| by Q or -Q depending
            # on the sign at qcut.
            Qloc = DFTK.psp_local_polynomial(T, element.psp)
            qcut = sqrt(2Ecut2)
            rts = [real.(rt) for rt in roots(Qloc) if imag(rt) < 1e-10]
            @assert length(rts) == 0 || maximum(rts) < qcut
            sign(Qloc(qcut)) < 0 && (Qloc = -Qloc)
            R = bound_sum_polynomial_gaussian(Qloc, model.recip_lattice, qcut,
                                              β=element.psp.rloc, k=-2, σ=2)
        else
            @assert element isa ElementCohenBergstresser
            R = 0  # assume Ecut2 to be large enough
        end
        accu += length(positions) * R / model.unit_cell_volume
    end

    # For the remaining term (inside B2 ∩ B1^⟂) we use an estimate precomputed
    # with the function bound_potential_b2_Vloc_b2_infty at a very large value
    # of Ecut for the employed basis in the hamblock. This yields a value for
    # |P_B V P_B |_\intfy norm for a basis B which is guaranteed to be larger than
    # B2 and thus an upper bound to the term |P_{B2} V P_{B2} |_\intfy.
    #
    # NOTE: These values are specific to the lattice and atomic positions
    #       we employ here and are not transferable.
    @assert basis2.Ecut ≤ 5000  # Values precomputed at Ecut=5000
    if is_cohen_bergstresser(model)
        bound_Vinf = 0.6681312576847188
    elseif is_linear_atomic(model)
        bound_Vinf = 7.230556115103383
    else
        error("Not implemented")
    end

    accu + bound_Vinf
end

@doc raw"""
Compute ``\|P_{B} Vloc P_{B}\|_\infty`` for ``B`` being the basis used in `hamblock`.
"""
function bound_potential_b2_Vloc_b2_infty(hamblock)
    basis = hamblock.basis
    model = basis.model
    T = eltype(basis)
    idx_local = only(findall(t -> t isa AtomicLocal, model.term_types))

    # We compute the supremum of the real-space values we know
    # (shifting the potential implicitly, which we know does not change the gap)
    Vloc_extrema = extrema(hamblock.operators[idx_local].potential)
    extent = (Vloc_extrema[2] - Vloc_extrema[1]) / T(2)

    # Then we use a gradient correction for the fact that we do not know the values
    # inside the cell. If g is the gradient vector and δ the cell diameter than the
    # possible addition is <g|d>/2 ≤ ||g|| δ / 2 where d is any lattice vector.
    # In turn we estimate ||g||_2 ≤ ||g||_∞ = ||\hat{g}||_1 i.e. by the l1-norm of
    # the Fourier coefficients of the gradient, which is |G| times the Fourier coefficients
    # of the potential itself.

    # local_potential_fourier is the actual FT of the real-space potential
    potterm(el, r, Gcart) = Complex{T}(DFTK.local_potential_fourier(el, norm(Gcart))
                                       * cis(-dot(Gcart, model.lattice * r)))

    # sqrt(Ω) because of normalised basis used in DFTK
    pot(G) = sum(potterm(elem, r, model.recip_lattice * G) / sqrt(model.unit_cell_volume)
                 for (elem, positions) in model.atoms
                 for r in positions)

    # Another sqrt(Ω) from going to the l1-norm of the Fourier coefficients
    sqrtΩ = sqrt(model.unit_cell_volume)
    sumVderiv = sum(norm(model.recip_lattice * G) * abs(pot(G)) for G in G_vectors(basis))
    diameter = norm(model.lattice * 1 ./ T.(basis.fft_size))
    derivative_term = sumVderiv * diameter / T(2) / sqrtΩ

    extent + derivative_term
end


function bound_potential_perpb1_Vnloc(hamblock, hamblock_Ecut2)
    basis2 = hamblock_Ecut2.basis
    kpoint2 = hamblock_Ecut2.kpoint
    model = basis2.model
    Ecut2 = basis2.Ecut
    kpoint = hamblock.kpoint
    T = eltype(basis2)
    recip_lattice = model.recip_lattice
    @assert any(t isa AtomicLocal for t in model.term_types)
    @assert kpoint.coordinate == kpoint2.coordinate

    # Tabulated values for the L∞ norm of the spherical harmonics Ylm
    Ylm∞ = (sqrt(1 / 4T(π)), sqrt(3 / 4T(π)))  # My guess: (5 / 4π, 7 / 4π)

    # Norms of the G vectors in the Ecut2 basis, ignoring the DC
    Gnorms_Ecut2 = [norm(model.recip_lattice * (G + kpoint.coordinate))
                    for G in G_vectors(kpoint2)]

    # Norms of the G vectors only in Ecut2 basis, but not in Ecut basis
    Gs_complement = setdiff(G_vectors(kpoint2), G_vectors(kpoint))
    Gnorms_complement = [norm(model.recip_lattice * (G + kpoint.coordinate))
                         for G in Gs_complement]

    function norm_projectors(psp, i, l, Nli, Gnorms)
        psp_radials = eval_psp_projection_radial.(psp, i, l, Gnorms)
        sqrt(Nli + sum(abs2, psp_radials)) * Ylm∞[l + 1] / sqrt(model.unit_cell_volume)
    end

    function projector_bound(psp, l)
        @assert psp isa PspHgh
        n_proj_l = size(psp.h[l + 1], 1)

        sum_proj_Ecut2 = zeros(T, n_proj_l)
        sum_proj_complement = zeros(T, n_proj_l)

        for i in 1:n_proj_l
            # Bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) where t = psp.rp[l+1] * |G|.
            Qnloc = DFTK.psp_projection_radial_polynomial(T, psp, i, l)
            qcut = sqrt(2Ecut2) - norm(recip_lattice * kpoint2.coordinate)
            Nli = bound_sum_polynomial_gaussian(Qnloc * Qnloc, recip_lattice,
                                                qcut, β=psp.rp[l + 1])

            sum_proj_Ecut2[i] = norm_projectors(psp, i, l, Nli, Gnorms_Ecut2)
            sum_proj_complement[i] = norm_projectors(psp, i, l, Nli, Gnorms_complement)
        end

        sum_proj_Ecut2' * abs.(psp.h[l + 1]) * sum_proj_complement
    end

    @assert all(element isa ElementPsp for (element, _) in model.atoms)
    sum(maximum(projector_bound.(element.psp, 0:element.psp.lmax)) * length(positions)
        for (element, positions) in model.atoms)
end


@doc raw"""
Estimates a (rough) upper bound for ``\|V φ\|`` for any ``φ`` outside `Ecut`
and `V` are the potential terms of the hamiltonian `hamblock` given
on the `Ecut2` basis.

It is assumed that potential terms are decaying functions in Fourier
space beyond `Ecut2`.

hamblock:        Hamiltonian k-block in Ecut basis
hamblock_Ecut2:  Hamiltonian k-block in Ecut2 basis
"""
function bound_potential_perpb1(hamblock, hamblock_Ecut2)
    @assert hamblock_Ecut2.basis.Ecut > hamblock.basis.Ecut
    @assert hamblock_Ecut2.kpoint.coordinate == hamblock.kpoint.coordinate
    @assert hamblock.basis.model == hamblock_Ecut2.basis.model
    basis2 = hamblock_Ecut2.basis

    if is_cohen_bergstresser(basis2.model)
        @assert hamblock_Ecut2.basis.Ecut ≥ minimal_Ecut2(hamblock.basis)
        (AtomicLocal=bound_potential_perpb1_Vloc(hamblock, hamblock_Ecut2), )
    elseif is_linear_atomic(basis2.model)
        (AtomicLocal=bound_potential_perpb1_Vloc(hamblock, hamblock_Ecut2),
         AtomicNonlocal=bound_potential_perpb1_Vnloc(hamblock, hamblock_Ecut2))
    else
        error("Not implemented")
    end
end


###
### Estimating eigenvalue gaps
###

"""
Estimate the gap using a Schur complement.

hamblock:        Hamiltonian k-block in Ecut basis
hamblock_Ecut2:  Hamiltonian k-block in Ecut2 basis
λtilde:          Approximate eigenvalues of hamblock for which gaps to the respective
                 next eigenvalues are estimated.
ψk_Ecut2:        Eigenvectors for this k-point in Ecut2 basis
ngap:            Number of eigenvalues for which to estimate the gap
                 (Default: length(λtilde) - 1)
fast:            Use a slightly faster algorithm relying on (λtilde, ψk_Ecut2) being
                 the exact lower end of the spectrum of hamblock.
"""
function estimate_gaps_schur(hamblock, hamblock_Ecut2, λtilde, ψk_Ecut2; fast=false,
                             ngap=length(λtilde) - 1, λtol=1e-6, full_block=false)
    # Use a Schur complement to estimate the gap. If fast is true gap might not be exact.
    #
    # ham is the Hamiltonian in the Ecut basis we use.
    # This allows to decompose the (exact) shifted Hamiltonian into blocks:
    #     H = ( ham   W       )
    #         ( W†    hamperp )
    # where in a Fourier basis W only depends on the potential V. The eigenvalues
    # of this operator are
    #     λ1 ≤ λ2 ≤ ... ≤ λn
    # An upper bound to λ1 is λtilde[1]. To estimate the gap δ1 between λ2 and λ1
    # we need a lower bound to λ2. This can be estimated by that root
    # of the function
    #    f(μ) = ||Vφ||^2 / min(|λtilde .- μ|) + Ecut - ||Vφ|| - μ
    # which is closest to λtilde[2]. In this φ is normalised and has only support
    # beyond Ecut). This also assumes that the eigenvalues are all of single
    # multiplicity.
    #
    # Denoting this root as μ[1] an estimate for the gap of each eigenvalue
    # is the difference (μ[1] - λtilde[1])
    @assert hamblock_Ecut2.basis.Ecut > hamblock.basis.Ecut
    @assert hamblock_Ecut2.kpoint.coordinate == hamblock.kpoint.coordinate
    @assert ngap ≤ length(λtilde) - 1
    @assert length(λtilde) == size(ψk_Ecut2, 2)
    @assert length(G_vectors(hamblock_Ecut2.kpoint)) == size(ψk_Ecut2, 1)

    basis = hamblock.basis
    T = eltype(basis)
    Ecut = basis.Ecut
    nbas = size(hamblock, 2)
    gap = zero(λtilde)
    M = length(λtilde) - 1

    if full_block
        @timeit to "Full diagonalisation with evecs" begin
            # Don't trust anything. Fully diagonalise hamblock:
            λtilde, Xtest = eigen!(Hermitian(Array(hamblock)))
            @assert length(basis.kpoints) == 1  # Only implemented for 1-kpt case!
            ψk_Ecut2 = DFTK.interpolate_blochwave([Xtest], hamblock.basis,
                                                   hamblock_Ecut2.basis)[1]
            Xtest = nothing
            M = length(λtilde)
        end
    elseif !fast
        # Check approximation for eigenpairs is actually correct
        if T in (Float32, Float64)
            @timeit to "Partial diagonalisation" begin
                λtest = eigvals(Hermitian(Array(hamblock)), 1:length(λtilde) + 15)
                λtest = λtest[1:length(λtilde)]
            end
        else
            @timeit to "Full diagonalisation" begin
                λtest = eigvals(Hermitian(Array(hamblock)))[1:length(λtilde)]
            end
        end
        @assert maximum(abs, λtilde - λtest) < max(10λtol, 1e-16)
    end

    @timeit to "Compute terms estimating potential coupling of B_1 and B_1^⟂" begin
        # Compute upper bound for ||P_{B_1} V P_{B_1^⟂} ||_op  (just use ||V P_{B_1^⟂}||_op)
        Vop_b1_b1perp = sum(bound_potential_perpb1(hamblock, hamblock_Ecut2))

        # Compute upper bound for || P_{B_1^⟂} V P_{B_1^⟂} ||_op  (just use ||V P_{B_1^⟂}||_op)
        Vop_b1perp_b1perp = Vop_b1_b1perp

        # Compute upper bound for || P_B_2^⟂ V P_{B_1} ||_op
        Vop_b1_b2perp = sum(bound_ham_perpb2_b1(hamblock, hamblock_Ecut2.basis.Ecut))

        # Compute P_{B_1^⟂ ∩ B_2} H P_{B_1} ψk_Ecut2
        Hψk = apply_hamiltonian_perpb1capb2(hamblock, hamblock_Ecut2,
                                            @view ψk_Ecut2[:, 1:M])
    end

    for iλ in 1:ngap
        λn = λtilde[iλ]
        λnext = λtilde[iλ + 1]

        @timeit to "QR factorisation of Hψk_iλ" begin
            Hψk_iλ = @view Hψk[:, iλ+1:M]
            HψkR = qr(Hψk_iλ).R
            @assert size(HψkR, 1) == length(iλ+1:M)
            @assert size(HψkR, 2) == length(iλ+1:M)
            Λ = @view λtilde[iλ+1:M]
        end

        @timeit to "Bisection to find best μ" begin
            # The condition for the best μ is S_μ == 0 or (in our approximation)
            #   0 == Ecut - ||P_{B_1^⟂} V P_{B_1^⟂}||_op - μ
            #        - ||R (Λ - μI)^{-1} R'||_op
            #        - 2 ||R (Λ - μI)^{-1}||_op ||P_{B_1} V P_{B_2^⟂}||_op
            #        - ||P_{B_1} V P_{B_2^⟂}||_op^2 / (λ_{iλ+1} - μ)
            #        - ||P_{B_1} V P_{B_1^⟂}||_op^2 / (λ_{M+1} - μ)
            function Sμ(μn)
                accu = Ecut - Vop_b1perp_b1perp - μn
                accu += -opnorm(HψkR * Diagonal(1 ./ (Λ .- μn)) * HψkR')
                accu += -2opnorm(HψkR * Diagonal(1 ./ (Λ .- μn))) * Vop_b1_b2perp
                accu += -Vop_b1_b2perp^2 / (λnext - μn)
                if M != nbas
                    accu += -Vop_b1_b1perp^2 / (λtilde[M+1] - μn)
                end
                accu
            end

            # Use Bisection to find the best μ
            ε = max(λtol, 1000eps(T))
            μn = λn
            if sign(Sμ(λn + ε)) < 0
                # If this happens, we cannot find a suitable zero => zero gap
            elseif sign(Sμ(λnext - ε)) < 0
                try
                    μn = find_zero(Sμ, (λn + ε, λnext - ε), Bisection(), xatol=ε)
                    μn = max(λn, μn - ε)  # To be on the safe side
                catch e
                    e isa Roots.ConvergenceFailed || rethrow()
                end
            else  # sign(Sμ(λnext - ε)) ≥ 0 && sign(Sμ(λn + ε)) ≥ 0
                μn = λnext - ε  # Stick to λnext - ε
            end

            # Check that `μn` is in [λtilde[iλ], λtilde[iλ+1]]
            # and gives rise to an (almost) positive Sμ:
            if !(λn ≤ μn ≤ λnext) || !isfinite(Sμ(μn)) || Sμ(μn) < -ε
                gap[iλ] = 0.0
                μn = λtilde[iλ]
            else
                gap[iλ] = μn - λn
            end
        end
    end

    gap
end

###
### Model runners producing data
###

function run_model(model; Ecut=10, tol=1e-6, n_bands=8, only_gamma=false,
                   kline_density=10, kwargs...)
    basis = PlaneWaveBasis(model, Ecut)
    ham = Hamiltonian(basis)

    # Diagonalise (at the Gamma point) to get εF
    @timeit to "Diagonalise Gamma point" begin
        ham = Hamiltonian(basis)
        eigensolver(args...; kwargs...) = DFTK.lobpcg_hyper(args...; kwargs...,
                                                            display_progress=true)
        eigres = diagonalise_all_kblocks(eigensolver, ham, n_bands; tol=tol, kwargs...)
        εF = find_fermi_level(basis, eigres.λ)
    end

    ret_common = (basis=basis, εF=εF, Ecut=Ecut, ham=ham, tol=tol)
    only_gamma && return merge(eigres, ret_common)

    # Compute kpath and band data
    @timeit to "Compute bands" begin
        kpath = DFTK.high_symmetry_kpath(basis.model; kline_density=kline_density)
        ρ0 = guess_density(basis)
        data = DFTK.compute_bands(basis, ρ0, kpath.kcoords, n_bands, tol=tol)
    end

    merge(ret_common, data, (kpath=kpath, ))
end


function model_cohen_bergstresser(;T=Float64)
    Si = ElementCohenBergstresser(:Si)
    a = Si.lattice_constant
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    Model(Matrix{T}(lattice); atoms=atoms, terms=[Kinetic(), AtomicLocal()])
end

function model_linear_silicon(;T=Float64)
    a = 10.263141334305942  # Silicon lattice constant in Bohr
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    model_atomic(lattice, atoms)
end


function run_cohen_bergstresser(;T=Float64, kwargs...)
    data = run_model(model_cohen_bergstresser(T=T); kwargs...)
    merge(data, (system="Cohen-Bergstresser", ))
end

function run_linear_silicon(;T=Float64, kwargs...)
    data = run_model(model_linear_silicon(T=T); kwargs...)
    merge(data, (system="Linear Silicon", ))
end


###
### Error estimator functions for data
###

"""
Estimate the discretisation error for the passed data.
Multiple error estimates are used (Bauer-Fike and Kato-Temple).
Ecut2 is the Ecut for the bigger grid used for the residual computation
If nothing, a value will be derived automatically.
If `fast` is true a faster, but not necessarily exact algorithm is chosen to
estimate the eigenvalue gap.
"""
function estimate_discretisation_error(data; Ecut2=nothing, fast=false,
                                       ngap=nothing, full_block=false)
    @timeit to "Estimate discretisation error" begin
        basis = data.basis
        minEcut2 = minimal_Ecut2(basis)

        if isnothing(Ecut2)
            if is_cohen_bergstresser(basis.model)
                Ecut2 = ceil(minEcut2)
            else
                # Make the Ecut2 a bit larger than the minimal value needed
                # TODO No clue if this is the best way to do it, but it works for us.
                # Maybe it makes more sense to add a constant onto the minimal_Ecut2?
                Ecut2 = ceil(4minEcut2)
            end
        end

        # Compute part of the residual in bigger grid
        @timeit to "Compute residual part on Ecut2" begin
            kcoords = [kpt.coordinate for kpt in basis.kpoints]
            basis_big = PlaneWaveBasis(basis.model, Ecut2, kcoords, basis.ksymops)
            X_big = DFTK.interpolate_blochwave(data.X, basis, basis_big)
            ham_big = Hamiltonian(basis_big)
            # TEMPORARY
            bound_potential_b2_Vloc_b2_infty(ham_big.blocks[1])
            residual_Ecut2 = [ham_big.blocks[ik] * X_big[ik] - X_big[ik] * Diagonal(data.λ[ik])
                              for ik in 1:length(basis.kpoints)]
            residual_norm_Ecut2 = [norm.(eachcol(resk)) for resk in residual_Ecut2]
        end

        # Compute upper bound for the part of the residual we miss
        @timeit to "Estimate residual part beyond Ecut2" begin
            ham = Hamiltonian(basis)
            residual_norm_terms = [[bound_perpb2_residual(ham.blocks[ik], ψnk, Ecut2)
                                    for ψnk in eachcol(data.X[ik])]
                                   for ik in 1:length(basis.kpoints)]
            n_bands = length(residual_norm_terms[1])
            residual_norm_terms = Dict(
                key => [[residual_norm_terms[ik][iband][key] for iband in 1:n_bands]
                        for ik in 1:length(basis.kpoints)]
                for key in keys(residual_norm_terms[1][1])
            )
            residual_norm_terms = merge((computed=residual_norm_Ecut2, ),
                                        (; residual_norm_terms...))
            residual_norm = sum(residual_norm_terms)
        end

        # Estimate eigenvalue gaps
        @timeit to "Estimate gap (fast = $fast)" begin
            # gap[i] = estimated gap between λ[i] and λ[i+1]
            # mingap[i] = min(gap[i-1], gap[i])

            ham = Hamiltonian(data.basis)
            gap = [estimate_gaps_schur(ham.blocks[ik], ham_big.blocks[ik],
                                       data.λ[ik], X_big[ik], fast=fast,
                                       λtol=data.tol, full_block=full_block)
                   for ik in 1:length(basis.kpoints)]

            mingap = deepcopy(gap)
            for ik = 1:length(basis.kpoints)
                mingap[ik][1] = gap[ik][1]
                for iband = 2:length(gap[ik])
                    mingap[ik][iband] = min(gap[ik][iband - 1], gap[ik][iband])
                end
            end
        end

        # Bauer-Fike error estimate
        bound_bauer_fike = residual_norm

        # Kato-Temple error estimate
        bound_kato_temple = [residual_norm[ik].^2 ./ mingap[ik]
                             for ik in 1:length(basis.kpoints)]

        (gap=gap, mingap=mingap, bound_kato_temple=bound_kato_temple, Ecut2=Ecut2,
         bound_bauer_fike=bound_bauer_fike, residual_norm=residual_norm,
         residual_Ecut2=residual_Ecut2, residual_norm_terms=residual_norm_terms,
         exact_bounds=!fast, minEcut2=minEcut2)
    end
end

function estimate_arithmetic_error(data; n_bands=nothing)
    @timeit to "Estimate arithmetic error" begin
        T = eltype(data.basis)
        IntT = Interval{T}
        modelT = data.basis.model

        if isnothing(n_bands)
            n_bands = length(data.λ[1])
        end

        println("Interval-Hamiltonian setup ... this will take some time ...")
        @timeit to  "Hamiltonian setup" begin
            # TODO Would need to copy temperature, spin_polarisation etc ...
            model = Model(Matrix{IntT}(modelT.lattice); atoms=modelT.atoms, terms=modelT.term_types)
            kcoords = [kpt.coordinate for kpt in data.basis.kpoints]
            basis = PlaneWaveBasis(model, data.Ecut, kcoords, data.basis.ksymops,
                                   data.basis.kweights)
            ham = Hamiltonian(basis)
        end
        println("Interval-Hamiltonian setup done!")

        @timeit to "Convert eigenpairs to intervals"  begin
            # Get eigenvalues / eigenvector coefficients as intervals
            eigvals = [Array{IntT}(λk[1:n_bands]) for λk in data.λ]
            eigvecs = [Array{Complex{IntT}}(ψk[:, 1:n_bands]) for ψk in data.X]
        end

        @timeit to  "Compute residuals in IntervalArithmetic" begin
            residual_norm = similar(eigvals)
            @showprogress 0.5 "Compute Interval residuals: " for ik in 1:length(basis.kpoints)
                # Form Ritz values via Rayleigh quotient in interval arithmetic
                Λks = eigvecs[ik]' * (ham.blocks[ik] * eigvecs[ik])
                residual_norm[ik] = norm.(eachcol(Λks - Diagonal(eigvals[ik])))
            end
        end

        # The error of the algorithm (e.g. not iterated till full numerical convergence)
        # is (roughly) the midpoint of the residual error. This is not fully correct
        # as the interval arithmetic might capture errors unevenly: E.g. the residual by
        # the numerical procedure could be 1e-10 and interval arithmetic could report the
        # interval [0 1e-8] in which case this routine would give 5e-7, whereas the
        # actual answer is 1e-10. This fine because we will only deal with cases
        # where the numerical error is *much* smaller than the residual error,
        # such that in practice only intervals like [1e-10 - 1e-14, 1e-10 + 1e-14]
        # will result here. But this should be changed for a more general implementation.
        error_algorithm = [mid.(residual_norm[ik]) for ik in 1:length(basis.kpoints)]

        # The difference between the supremum of the residual_norm and the
        # error_algorithm is the arithmetic error. This is also the radius of the interval.
        error_arithmetic = [radius.(residual_norm[ik]) for ik in 1:length(basis.kpoints)]

        if n_bands != length(data.λ[1])
            for ik in 1:length(basis.kpoints)
                append!(error_algorithm[ik], zeros(length(data.λ[1]) - n_bands))
                append!(error_arithmetic[ik], zeros(length(data.λ[1]) - n_bands))
            end
        end
        (error_algorithm=error_algorithm, error_arithmetic=error_arithmetic)
    end
end


###
### Storing results
###

function store_results(h5name, data; T=Float64, subdir=nothing, n_bands=nothing)
    isfile(h5name) || close(h5open(h5name, "w"))

    if isnothing(n_bands)  # Determine number of bands to store
        n_bands = length(data.λ[1])
    end

    h5open(h5name, "r+") do h5f
        if isnothing(subdir)
            h5gr = h5f
        elseif exists(h5f, subdir)
            h5gr = g_open(h5f, subdir)
        else
            h5gr = g_create(h5f, subdir)
        end

        if exists(h5gr, "System")
            @assert read(h5gr, "System") == data.system
        else
            h5gr["System"] = data.system
        end

        Ecut = round(data.Ecut, digits=3)
        if exists(h5gr, "Ecuts")
            Ecuts = append!(h5gr["Ecuts"][:], Ecut)
            o_delete(h5gr, "Ecuts")
            h5gr["Ecuts"] = Ecuts
        else
            h5gr["Ecuts"] = Float64[Ecut]
        end

        sEcut = (@sprintf "%.3f" Ecut)
        if hasproperty(data, :kpath)
            for label in keys(data.kpath.klabels)
                h5gr["$sEcut/klabels/$label"] = data.kpath.klabels[label]
            end
        end
        h5gr["$sEcut/n_kpoints"] = length(data.basis.kpoints)

        for symbol_float in (:Ecut2, :tol, :Ecut, :εF, :minEcut2)
            if hasproperty(data, symbol_float)
                h5gr["$sEcut/" * string(symbol_float)] =  T(getproperty(data, symbol_float))
            end
        end
        for symbol in (:exact_bounds, )
            if hasproperty(data, symbol)
                h5gr["$sEcut/" * string(symbol)] = getproperty(data, symbol)
            end
        end

        for ik in 1:length(data.basis.kpoints)
            exists(h5gr, "$sEcut/kpt_$ik") && o_delete(h5gr, "$sEcut/kpt_$ik")
            h5kpt = g_create(h5gr, "$sEcut/kpt_$ik")

            h5kpt["coordinate"] = Vector{T}(data.basis.kpoints[ik].coordinate)

            if hasproperty(data, :λ)
                h5kpt["eigenvalues"] = Array{T}(data.λ[ik][1:n_bands])
                h5kpt["eigenvalues_string"] = join(string.(data.λ[ik][1:n_bands]), ",")
            end

            for symbol in (:gap, :mingap, :bound_kato_temple, :bound_bauer_fike,
                           :residual_norm, :error_algorithm, :error_arithmetic)
                if hasproperty(data, symbol)
                    h5kpt[string(symbol)] = (
                        Array{T}(getproperty(data, symbol)[ik][1:n_bands])
                    )
                end
            end

            if hasproperty(data, :residual_norm_terms)
                for key in keys(data.residual_norm_terms)
                    h5kpt["residual_norm_term_" * string(key)] = (
                        Array{T}(getproperty(data.residual_norm_terms, key)[ik][1:n_bands])
                    )
                end
            end
        end
    end

    nothing
end

function load_results(h5name; has_subdirs=false)
    function parse_subdir(h5gr)
        system = read(h5gr, "System")
        Ecuts = read(h5gr, "Ecuts")

        results = []
        for Ecut in Ecuts
            sEcut = @sprintf "%.3f" Ecut
            result = Dict{Symbol, Any}()
            result[:n_kpoints] = read(h5gr, "$sEcut/n_kpoints")

            if "klabels" in names(h5gr["$sEcut"])
                result[:klabels] = Dict(label => h5gr["$sEcut/klabels/$label"][:]
                                        for label in names(h5gr["$sEcut/klabels"]))
            end
            for symbol in (:Ecut_big, :exact_bounds, :tol, :Ecut, :εF, :Ecut2, :minEcut2)
                if string(symbol) in names(h5gr["$sEcut"])
                    result[symbol] = read(h5gr, "$sEcut/" * string(symbol))
                end
            end

            ksymbols = Vector{Symbol}()
            for symbol in (:gap, :mingap, :bound_kato_temple, :bound_bauer_fike,
                           :residual_norm, :error_algorithm, :error_arithmetic,
                           :residual_norm_term_computed,
                           :residual_norm_term_AtomicLocal,
                           :residual_norm_term_AtomicNonlocal)
                if string(symbol) in names(h5gr["$sEcut/kpt_1"])
                    push!(ksymbols, symbol)
                    result[symbol] = Vector{Vector{Float64}}()
                end
            end

            result[:λ] = Vector{Vector{Double64}}()
            result[:kcoords] = Vector{Vector{Float64}}()
            for ik in 1:result[:n_kpoints]
                h5kpt = h5gr["$sEcut/kpt_$ik"]

                push!(result[:kcoords], read(h5kpt, "coordinate"))
                push!(result[:λ], parse.(Double64,
                                         split(read(h5kpt, "eigenvalues_string"), ",")))
                for symbol in ksymbols
                    push!(result[symbol], read(h5kpt, string(symbol)))
                end
            end
            push!(results, result)
        end
        (Ecuts=Ecuts, results=results, system=system)
    end

    h5open(h5name, "r") do h5f
        if has_subdirs
            results = Dict{String, Any}()
            for subdir in names(h5f)
                results[subdir] = parse_subdir(h5f[subdir])
            end
            return results
        else
            return parse_subdir(h5f)
        end
    end
end

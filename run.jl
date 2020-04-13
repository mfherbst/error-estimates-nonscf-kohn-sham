include("common.jl")

# Produce the data needed for the plots of the paper

###
### Cohen-Bergstresser
###

function cb_gamma_versus_Ecut_Double64(;n_bands=8)
    reset_timer!(to)

    h5name = "cb_gamma_Ecut_$(n_bands)bands_Double64_fast_gap.hdf5"
    isfile(h5name) && error("Remove $h5name first")
    for Ecut in append!(collect(3:19), collect(20:2:70))
        println("#\n#-- Ecut = $Ecut\n#")
        tol = 1e-28
        Ecut < 35 && (tol = 1e-25)
        Ecut < 22 && (tol = 5e-17)

        fast = false
        (Ecut > 55) && (fast = true)

        data = run_cohen_bergstresser(Ecut=Ecut, T=Double64, tol=tol, only_gamma=true,
                                      n_bands=n_bands)
        errors1 = estimate_discretisation_error(data, fast=fast)
        store_results(h5name, merge(data, errors1))
    end

    display(to)
end

function cb_gamma_versus_Ecut_Float64(;n_bands=8)
    reset_timer!(to)

    h5name = "cb_gamma_Ecut_$(n_bands)bands_Float64.hdf5"
    isfile(h5name) && error("Remove $h5name first")
    for Ecut in collect(3:70)
        println("#\n#-- Ecut = $Ecut\n#")
        tol = 1e-10
        Ecut > 12 && (tol = 1e-12)

        data = run_cohen_bergstresser(Ecut=Ecut, T=Float64, tol=tol, only_gamma=true,
                                      n_bands=n_bands)
        errors1 = estimate_discretisation_error(data, fast=false)
        errors2 = estimate_arithmetic_error(data)
        store_results(h5name, merge(data, errors1, errors2))
    end

    display(to)
end

function cb_bands(;Ecut=10, T=Float64, tol=6)
    reset_timer!(to)

    h5name = "cb_bands.hdf5"
    isfile(h5name) && error("Remove $h5name first")

    tol = 10.0^(-tol)
    data = run_cohen_bergstresser(Ecut=Ecut, T=T, tol=tol, kline_density=7)
    errors1 = estimate_discretisation_error(data, fast=true)
    errors2 = estimate_arithmetic_error(data)
    store_results(h5name, merge(data, errors1, errors2))

    display(to)
end


###
### Linear silicon
###

function si_gamma_versus_Ecut_full_bands_Float64(;maxEcut=36)
    reset_timer!(to)
    h5name = "si_gamma_Ecut_full_bands_Float64.hdf5"
    isfile(h5name) && error("Remove $h5name first")

    for Ecut in collect(6:2:maxEcut)
        println("#\n#-- Ecut = $Ecut\n#")
        tol = 1e-10
        Ecut > 45 && (tol = 5e-12)

        data = run_linear_silicon(Ecut=Ecut, T=Float64, tol=tol, only_gamma=true)
        errors1 = estimate_discretisation_error(data, full_block=true)
        store_results(h5name, merge(data, errors1))
    end

    display(to)
end

function si_gamma_versus_Ecut_Float64(;n_bands=25, maxEcut=100)
    reset_timer!(to)

    h5name = "si_gamma_Ecut_$(n_bands)bands_Float64.hdf5"
    isfile(h5name) && error("Remove $h5name first")
    for Ecut in collect(6:2:maxEcut)
        println("#\n#-- Ecut = $Ecut\n#")
        tol = 1e-10
        Ecut > 45 && (tol = 5e-12)

        data = run_linear_silicon(Ecut=Ecut, T=Float64, tol=tol, only_gamma=true,
                                  n_bands=n_bands)
        errors1 = estimate_discretisation_error(data, fast=false)
        store_results(h5name, merge(data, errors1))
    end

    display(to)
end
si_gamma_versus_Ecut_25_bands_Float64() = si_gamma_versus_Ecut_Float64(n_bands=25, maxEcut=100)
si_gamma_versus_Ecut_50_bands_Float64() = si_gamma_versus_Ecut_Float64(n_bands=50, maxEcut=60)

function si_gamma_residual_versus_Ecut2(;Ecut=20)
    reset_timer!(to)

    h5name = "si_gamma_residual_versus_Ecut2.hdf5"
    isfile(h5name) && error("Remove $h5name first")

    # Determine minimal Ecut2:
    basis = PlaneWaveBasis(model_linear_silicon(), Ecut)
    minEcut2 = ceil(minimal_Ecut2(basis))

    for Ecut2 in collect(minEcut2:100)
        sEcut2 = @sprintf "%.3f" Ecut2
        println("#\n#-- Ecut2 = $sEcut2\n#")
        data = run_linear_silicon(Ecut=Ecut, T=Float64, tol=1e-10, only_gamma=true)
        errors1 = estimate_discretisation_error(data, fast=false, Ecut2=Ecut2)
        store_results(h5name, merge(data, errors1), subdir=sEcut2)
    end

    display(to)
end


function si_bands(;Ecut=42, T=Float64, tol=6, n_bands=35)
    reset_timer!(to)

    h5name = "si_bands.hdf5"
    isfile(h5name) && error("Remove $h5name first")

    tol = 10.0^(-tol)
    data = run_linear_silicon(Ecut=Ecut, T=T, tol=tol, kline_density=7, n_bands=n_bands)
    errors1 = estimate_discretisation_error(data, fast=true)
    errors2 = estimate_arithmetic_error(data, n_bands=10)  # Otherwise too expensive
    store_results(h5name, merge(data, errors1, errors2), n_bands=10)

    display(to)
end


function main()
    functions = [
        cb_gamma_versus_Ecut_Double64,
        cb_gamma_versus_Ecut_Float64,
        cb_bands,
        #
        si_gamma_residual_versus_Ecut2,
        si_gamma_versus_Ecut_25_bands_Float64,
        si_gamma_versus_Ecut_50_bands_Float64,
        si_gamma_versus_Ecut_full_bands_Float64,
        si_bands,
    ]

    for fun in functions
        println("#\n#-- Running $fun\n#")
        fun()
        println("\n#\n#" * "-" ^ 70 * "\n#\n")
    end
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()

include("common.jl")
using Plots
import PGFPlots
pgfplots()
PGFPlots.pushPGFPlotsPreamble(raw"\usepackage{amsmath}")

# Conventions used for approx. eigenvalues and wavefunctions
const EPS = raw"\widetilde{\varepsilon}"
const PSI = raw"\widetilde{u}"

const defm = (3, :xcross, Plots.Stroke(1.0, :black, 0.8, :solid))

function gap_versus_ecut(data, data2=nothing, data3=nothing; iband=1, ik=1, log=false)
    Ecuts = data.Ecuts
    results = data.results
    n_bands = length(results[1][:λ][1])

    eigenvalues = [results[iEcut][:λ][ik][iband] for iEcut = 1:length(Ecuts)]
    eigenvalues_next = [results[iEcut][:λ][ik][iband+1]
                        for iEcut = 1:length(Ecuts)]
    estimated = [results[iEcut][:gap][ik][iband] for iEcut = 1:length(Ecuts)]
    computed = eigenvalues_next - eigenvalues

    kwargs = ()
    if log
        estimated = computed[end] .- estimated
        kwargs = (yaxis=:log, )
    end
    estlabel = (raw"$\mu^\ast_{" * "$(iband+1)" * raw"} - " *
                EPS * raw"_{" * "$iband" * raw",X}$")
    estlabel1 = isnothing(data2) ? estlabel : estlabel * " (\$M = $n_bands\$)"

    p = plot(legend=:bottomright; xlabel=raw"$E_\text{cut}$", kwargs...)
    if !log
        plot!(p, Ecuts, computed,
              label=(raw"$" * EPS * raw"_{" * "$(iband+1)" * raw",X} - " *
                     EPS * raw"_{" * "$iband" * raw",X}$"),
              m=defm, linestyle=:dot, color=2)
    end
    if !isnothing(data3)
        d3_Ecuts = data3.Ecuts
        @assert length(d3_Ecuts) ≤ length(Ecuts)
        n_bands3 = length(data3.results[1][:λ][1])
        estimated3 = [data3.results[iEcut][:gap][ik][iband]
                      for iEcut = 1:length(d3_Ecuts)]
        if log
            estimated3 = computed[end] .- estimated3
        end
        plot!(p, d3_Ecuts, estimated3, label=estlabel * raw" ($M = \text{dim}(X)$)", m=defm,
              linestyle=:dashdot, color=4)
    end
    if !isnothing(data2)
        d2_Ecuts = data2.Ecuts
        @assert length(d2_Ecuts) ≤ length(Ecuts)
        n_bands2 = length(data2.results[1][:λ][1])
        estimated2 = [data2.results[iEcut][:gap][ik][iband]
                      for iEcut = 1:length(d2_Ecuts)]
        if log
            estimated2 = computed[end] .- estimated2
        end
        plot!(p, d2_Ecuts, estimated2, label=estlabel * " (\$M = $n_bands2\$)", m=defm,
              linestyle=:dash, color=3)
    end
    plot!(p, Ecuts, estimated, label=estlabel1,
          m=defm, linestyle=:solid, color=1)

    p
end

function residual_total_versus_ecut(data; selected_bands=[1], ik=1, arithmetic=true)
    Ecuts = data.Ecuts
    results = data.results
    n_bands = length(results[1][:λ][ik])

    # Prepare arrays
    residual_estimated = [[results[iEcut][:residual_norm][ik][iband]
                           for iEcut = 1:length(Ecuts)] for iband = 1:n_bands]

    # Note: Algorithm error is contained inside residual_computed already
    error_algorithm = [[results[iEcut][:error_algorithm][ik][iband]
                        for iEcut = 1:length(Ecuts)] for iband = 1:n_bands]
    error_arithmetic = [[results[iEcut][:error_arithmetic][ik][iband]
                        for iEcut = 1:length(Ecuts)] for iband = 1:n_bands]
    error_algoarith = [error_arithmetic[iband] + error_algorithm[iband]
                       for iband in 1:n_bands]

    p = plot(legend=:topright, xlabel=raw"$E_\text{cut}$", yaxis=:log)
    for iband in selected_bands
        kwargs = (colour=iband, )
        siband = " $iband"
        if length(selected_bands) == 1
            kwargs = NamedTuple()
            siband = ""
        end
        arithmetic || (kwargs = merge(kwargs, (m=defm, )))
        plot!(p, Ecuts, residual_estimated[iband];
              label=(raw"$\|V " * PSI * raw"\|$" * siband), kwargs...)
        if arithmetic
            plot!(p, Ecuts, error_algoarith[iband]; ls=:dash,
                  label="arithmetic + algorithm error" * siband, kwargs...)
        end
    end
    p
end


function residual_versus_ecut(data; selected_bands=[1], ik=1)
    Ecuts = data.Ecuts
    results = data.results
    n_bands = length(results[1][:λ][ik])

    # Prepare arrays
    residual_estimated = [[results[iEcut][:residual_norm][ik][iband]
                           for iEcut = 1:length(Ecuts)] for iband = 1:n_bands]
    residual_computed = [[results[iEcut][:residual_norm_term_computed][ik][iband]
                          for iEcut = 1:length(Ecuts)] for iband = 1:n_bands]
    residual_loc = [[results[iEcut][:residual_norm_term_AtomicLocal][ik][iband]
                     for iEcut = 1:length(Ecuts)] for iband = 1:n_bands]
    residual_nloc = [[results[iEcut][:residual_norm_term_AtomicNonlocal][ik][iband]
                      for iEcut = 1:length(Ecuts)] for iband = 1:n_bands]

    # Derived arrays
    residual_beyond = [residual_loc[iband] .+ residual_nloc[iband] for iband = 1:n_bands]
    p = plot(legend=:bottomleft, xlabel=raw"$E_\text{cut}$", yaxis=:log)
    for iband in selected_bands
        getcolor(iband, i) = length(selected_bands) == 1 ? i : iband
        siband = " $iband"
        if length(selected_bands) == 1
            siband = ""
        end

        labels = [raw"$\|P_{Y} V " * PSI * raw"\|$" * siband,
                  raw"estimated $\|P_{Y^\perp} V " * PSI * raw"\|$" * siband,
                  raw"estimated $\|V " * PSI * raw"\|$" * siband]
        iband != selected_bands[1] && (labels[2:end] .= "")

        plot!(p, Ecuts, residual_estimated[iband]; label=labels[3],
              linestyle=:solid, color=getcolor(iband, 1))
        plot!(p, Ecuts, residual_computed[iband]; label=labels[1],
              linestyle=:dash, color=getcolor(iband, 2))
        plot!(p, Ecuts, residual_beyond[iband]; label=labels[2],
              linestyle=:dot, color=getcolor(iband, 3))
    end
    p
end

function residual_versus_ecut2(data::AbstractDict; iband=1, ik=1, iEcut=1)
    sEcuts = sort(collect(keys(data)), by=sEcut -> data[sEcut].results[iEcut][:Ecut2])
    @assert all(length(data[sEcut].results) == 1 for sEcut in sEcuts)

    # Prepare data arrays
    Ecut2s = [data[sEcut].results[iEcut][:Ecut2] for sEcut in sEcuts]
    residual_computed = [data[sEcut].results[iEcut][:residual_norm_term_computed][ik][iband]
                         for sEcut in sEcuts]
    residual_loc = [data[sEcut].results[iEcut][:residual_norm_term_AtomicLocal][ik][iband]
                    for sEcut in sEcuts]
    residual_nloc = [data[sEcut].results[iEcut][:residual_norm_term_AtomicNonlocal][ik][iband]
                     for sEcut in sEcuts]
    residual_estimated = [data[sEcut].results[iEcut][:residual_norm][ik][iband]
                          for sEcut in sEcuts]

    p = plot(legend=:bottomleft, xlabel=raw"$E_\text{cut}^{(2)}$", yaxis=:log)
    plot!(p, Ecut2s, residual_estimated,
          label=raw"estimated $\|V " * PSI * raw"\|$")
    plot!(p, Ecut2s, residual_computed;
          label=raw"$\| P_Y V " * PSI *  raw"\|$", ls=:dash)
    plot!(p, Ecut2s, residual_loc;
          label=raw"estimated $\|P_{Y^\perp} V_\text{loc} " * PSI *  raw"\|$", ls=:dot)
    plot!(p, Ecut2s, residual_nloc;
          label=raw"estimated $\|P_{Y^\perp} V_\text{nonloc} " * PSI * raw"\|$", ls=:dashdot)

    p
end

function energy_errors_versus_ecut(data; selected_bands=[1, 2], ik=1)
    results = data.results
    n_bands = length(results[1][:λ][ik])

    # Prepare arrays
    eigenvalues = [[results[iEcut][:λ][ik][iband] for iEcut = 1:length(data.Ecuts)]
                    for iband = 1:n_bands]
    bound_bauer_fike = [[results[iEcut][:bound_bauer_fike][ik][iband]
                         for iEcut = 1:length(data.Ecuts)] for iband = 1:n_bands]
    bound_kato_temple = [[results[iEcut][:bound_kato_temple][ik][iband]
                          for iEcut = 1:length(data.Ecuts)] for iband = 1:n_bands]

    Ecuts = data.Ecuts[1:end-1]
    p = plot(legend=:bottomleft, yaxis=:log, xlabel=raw"$E_\text{cut}$")
    for iband in selected_bands
        kwargs = (colour = iband, )
        siband = " $iband"
        if length(selected_bands) == 1
            kwargs = ()
            siband = ""
        end
        labels = ["``true'' error$siband", "Bauer-Fike$siband", "Kato-Temple$siband"]
        if iband != selected_bands[1]
            labels[2:end] .= ""
        end

        plot!(p, Ecuts, bound_bauer_fike[iband][1:end-1], label=labels[2],
              linestyle=:dash; kwargs...)

        kt_plot = copy(bound_kato_temple[iband][1:end-1])
        kt_plot[isinf.(kt_plot)] .= NaN
        plot!(p, Ecuts, kt_plot, label=labels[3], linestyle=:dot; kwargs...)

        error = abs.(eigenvalues[iband][1:end-1] .- eigenvalues[iband][end])
        error[iszero.(error)] .= NaN
        plot!(p, Ecuts, error, label=labels[1]; kwargs...)
    end
    p
end

function bands_errors(data; iEcut=1)
    results = data.results[iEcut]

    # Reconstruct a dummy basis
    if data.system == "Cohen-Bergstresser"
        model = model_cohen_bergstresser()
    elseif data.system == "Linear Silicon"
        model = model_linear_silicon()
    else
        error("Model $(data.system) not known.")
    end
    ksymops = [[(Mat3{Int}(I), Vec3(zeros(3)))] for _ in 1:results[:n_kpoints]]
    basis = PlaneWaveBasis(model, results[:Ecut], results[:kcoords], ksymops)

    # Compute the error to display on the bands
    results[:bound_best] = deepcopy(results[:bound_bauer_fike])
    results[:bound_best_label] = [fill("BF", length(results[:bound_bauer_fike][ik]))
                                  for ik in 1:results[:n_kpoints]]
    for ik in 1:results[:n_kpoints]
        for iband in 1:length(results[:bound_bauer_fike][ik])
            kt_lower = (results[:bound_kato_temple][ik][iband]
                        ≤ results[:bound_bauer_fike][ik][iband])
            if kt_lower
                results[:bound_best][ik][iband] = results[:bound_kato_temple][ik][iband]
                results[:bound_best_label][ik][iband] = "KT"
            end
        end
    end

    λerror = [(   results[:bound_best][ik]        # discretisation error
                + results[:error_algorithm][ik]   # algorithmic error
                + results[:error_arithmetic][ik]  # floating-point error
              ) for ik in 1:results[:n_kpoints]]

    shift = results[:εF]
    shift = results[:λ][1][4]  # Shift wrt. 4th eigenvalue at Gamma point
    plot_band_data_mod((basis=basis, λ=results[:λ], λerror=λerror),
                       results[:bound_best_label],
                       shift=shift, klabels=results[:klabels])
end

function plot_band_data_mod(band_data, best_label; shift=nothing,
                            klabels=Dict{String, Vector{Float64}}(), unit=:Ha)
    @assert !isnothing(shift)
    data = DFTK.prepare_band_data(band_data, klabels=klabels)

    Δ = 0.025  # length of error bars (in units of kdistance)

    kstart = 0
    # For each branch, plot all bands, spins and errors
    p = plot()
    for ibranch = 1:data.n_branches
        kdistances = data.kdistances[ibranch]
        for iband = 1:data.n_bands
            energies = (data.λ[ibranch][:up][iband, :] .- shift)
            yerror = data.λerror[ibranch][:up][iband, :]

            plot!(p, kdistances, energies, color=:black, label="")  # Bands

            redpoints_x = Float64[]
            redpoints_y = Float64[]
            bluepoints_x = Float64[]
            bluepoints_y = Float64[]
            for i in 1:length(kdistances)
                xs = [kdistances[i] - Δ, kdistances[i] + Δ, NaN,
                      kdistances[i] - Δ, kdistances[i] + Δ, NaN,
                      kdistances[i], kdistances[i], NaN]
                ys = [energies[i] - yerror[i], energies[i] - yerror[i], NaN,
                      energies[i] + yerror[i], energies[i] + yerror[i], NaN,
                      energies[i] - yerror[i], energies[i] + yerror[i], NaN]

                if best_label[kstart + i][iband] == "BF"
                    append!(bluepoints_x, xs)
                    append!(bluepoints_y, ys)
                else
                    append!(redpoints_x, xs)
                    append!(redpoints_y, ys)
                end
            end
            common = (linewidth=1, label="", linealpha=0.85)
            plot!(p, redpoints_x, redpoints_y; color=:red, common...)
            plot!(p, bluepoints_x, bluepoints_y; color=:blue, common...)
        end
        kstart += length(kdistances)
    end

    # X-range: 0 to last kdistance value
    xlims!(p, (0, data.kdistances[end][end]))
    xticks!(p, data.ticks["distance"], data.ticks["label"])
    ylims!(p, -0.2, 0.2)

    p
end

function setup()
    fac = 0.75
    default(size=tuple(Int.(ceil.(fac .* [600, 400]))...))
end


function main()
    setup()
    mticks(r) = 10. .^r, ["10^{$i}" for i in r]


    # Cohen-Bergstresser
    #
    begin
        cb_gamma = load_results("cb_gamma_Ecut_8bands_Float64.hdf5")
        p = gap_versus_ecut(cb_gamma)
        savefig(xlims!(p, -Inf, 15), "cb_gap_ecut.pdf")

        p = residual_total_versus_ecut(cb_gamma, selected_bands=[1], arithmetic=false)
        savefig(ylims!(xlims!(p, -Inf, 30), 1e-8, Inf), "cb_residual_versus_ecut.pdf")

        p = residual_total_versus_ecut(cb_gamma, selected_bands=[1], arithmetic=true)
        savefig(yticks!(p, mticks(0:-2:-13)...), "cb_residual_arithmetic_versus_ecut.pdf")
    end
    begin
        cb_gamma_Df64 = load_results("cb_gamma_Ecut_8bands_Double64_fast_gap.hdf5")
        savefig(energy_errors_versus_ecut(cb_gamma_Df64, selected_bands=[1]),
                "cb_energy_errors_versus_ecut.pdf")
    end
    begin
        cb_bands = load_results("cb_bands.hdf5")
        savefig(bands_errors(cb_bands), "cb_band_errors.pdf")
    end

    # Silicon
    #
    begin
        p = residual_versus_ecut2(load_results("si_gamma_residual_versus_Ecut2.hdf5",
                                               has_subdirs=true))
        savefig(p, "si_residual_versus_ecut2.pdf")
    end
    begin
        si_gamma25 = load_results("si_gamma_Ecut_25bands_Float64.hdf5")
        si_gamma50 = load_results("si_gamma_Ecut_50bands_Float64.hdf5")
        si_gamma_full = load_results("si_gamma_Ecut_full_bands_Float64.hdf5")
        p = gap_versus_ecut(si_gamma25, si_gamma50, si_gamma_full)
        savefig(xlims!(p, -Inf, 65), "si_gap_ecut.pdf")
        p = residual_versus_ecut(si_gamma25, selected_bands=[1])
        savefig(yticks!(p, mticks(0:-2:-14)...), "si_residual_versus_ecut.pdf")
        p = energy_errors_versus_ecut(si_gamma25, selected_bands=[1])
        savefig(yticks!(p, mticks(0:-2:-14)...), "si_energy_errors_versus_ecut.pdf")
    end
    begin
        si_bands = load_results("si_bands.hdf5")
        savefig(bands_errors(si_bands), "si_band_errors.pdf")
    end
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()

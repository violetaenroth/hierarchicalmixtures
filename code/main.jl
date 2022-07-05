using Random
using Distributions
using Statistics
using SpecialFunctions
using StatsPlots
using CSV
using DataFrames


include("functions.jl")

# function MCMC(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, T, T0)
#     I = length(y)
#     n = [length(y[i]) for i in 1:I]
#
#     # Initial configuration
#     c = [ones(Int64, n[j]) for j in 1:I]
#     d = [ones(Int64, 1) for j in 1:I]
#     l_i = ones(Int64, I)
#     l_d = [I]
#     q_bullet =[ones(Int64, 1).*n[j] for j in 1:I]
#     Kn = 1
#
#     # Outputs
#     c_out = [[[] for i in 1:I] for t in 1:T]
#     q_bullet_out = [[[] for i in 1:I] for t in 1:T]
#     d_out = [[[] for i in 1:I] for t in 1:T]
#     l_i_out = [[] for i in 1:T]
#     l_d_out = [[] for i in 1:T]
#     Kn_out = zeros(Int64, T)
#
#     # Output parameters
#     Mu_out = [[] for i in 1:T]
#     Sigma2_out = [[] for i in 1:T]
#
#     if(T0>0)
#         print("Burning-in \n")
#         for t in 1:T0
#             sim = iteration(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, c, d, l_i, l_d, q_bullet, Kn)
#             c = sim.c
#             d = sim.d
#             l_i = sim.l_i
#             l_d = sim.l_d
#             q_bullet = sim.q_bullet
#             Kn = sim.Kn
#         end
#     end
#     print("Sampling \n")
#     if (T>0)
#         for t in 1:T
#             sim = iteration(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, c, d, l_i, l_d, q_bullet, Kn)
#
#             c = sim.c
#             d = sim.d
#             l_i = sim.l_i
#             l_d = sim.l_d
#             q_bullet = sim.q_bullet
#             Kn = sim.Kn
#
#             # Saving the outputs
#             c_out[t] = sim.c
#             d_out[t] = sim.d
#             l_i_out[t] = sim.l_i #setindex!(l_i_out, sim.l_i, t)
#             l_d_out[t] = sim.l_d #setindex!(l_d_out, sim.l_d, t)
#             q_bullet_out[t] = sim.q_bullet
#             Kn_out[t] = sim.Kn
#
#             Mu_out[t] = sim.Mu
#             Sigma2_out[t] = sim.Sigma2
#             end
#     end
#     return (c = c_out, d = d_out, l_i = l_i_out, l_d = l_d_out, q_bullet = q_bullet_out, Kn = Kn_out, Mu = Mu_out, Sigma2 = Sigma2_out)
# end


#########
Random.seed!(1234)

mu_simul = [[-4 0 4.5], [0 -3.5]]
sigma2_simul = [[1 0.5 1.2], [0.5 1]]
# Components weights
weights_simul = [[0.6 0.2 0.2], [0.7 0.3]]

#Number of observations in each group
n = [100 100]

y = Matrix{Any}(undef, 1, I)

for j = 1:I
    samp = []
    for k = 1:length(weights_simul[j])
        append!(samp, rand(Normal(mu_simul[j][k], sigma2_simul[j][k]), Int(weights_simul[j][k]*n[j])))
    end
    y[j] = samp
end




T = 4000
T0 = 5000 #burn in iterations


mu0 = mean(vcat(y...))

# Stable process
sigma = 0.3253
sigma0 = 0.7406
theta = 0
theta0 = 0
tau0 = 0.668669
a = 3.59685
b = 8.68216

Stablemcmc =  MCMC(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, T, T0)
stableDensity = dens(Stablemcmc, [8, 2], [-8, -6], 100)
stableGroups = posteriorKn(Stablemcmc)

# Dirichlet process
sigma = 0
sigma0 = 0
theta = 0.9475
theta0 = 5.5837
tau0 = 0.47762
a = 4.23559
b = 10.8177

Dirichletmcmc = MCMC(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, T, T0)
dirichletDensity = dens(Dirichletmcmc, [8, 2], [-8, -6], 100)
dirichletGroups = posteriorKn(Dirichletmcmc)

#Pitman-Yor process
sigma = 0.32435
theta = 0.002412
sigma0 = 0.4155
theta0 = 0.3047
tau0 = 0.668669
a = 3.59685
b = 8.68216

pitmanYormcmc =  MCMC(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, T, T0)
pitmanYorDensity = dens(pitmanYormcmc, [8, 2], [-8, -6], 100)
pitmanYorGroups = posteriorKn(pitmanYormcmc)

# Dirichlet-PitmanYor
sigma = 0.2243# 0.3243 
sigma0 = 0
theta =  0.0024
theta0 = 4.728 #6.4431
tau0 = 0.47762
a = 3.59685
b = 8.68216

dirPitmanYormcmc =  MCMC(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, T, T0)
dirPitmanYorDensity = dens(dirPitmanYormcmc, [8, 2], [-8, -6], 100)
dirPitmanYorGroups = posteriorKn(dirPitmanYormcmc)

#Stable - PitmanYor
sigma = 0.3243
sigma0 = 0.7488
theta =  0.0024
theta0 = 0
tau0 = 0.47762
a = 4.23559
b =10.8177

StbPitmanYormcmc =  MCMC(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, T, T0)
StbPitmanYorDensity = dens(StbPitmanYormcmc, [8, 2], [-8, -6], 100)
StbPitmanYorGroups = posteriorKn(StbPitmanYormcmc)

# PitmanYor -Dirichlet
sigma = 0
sigma0 = 0.28869
theta =  0.9475
theta0 = 3.3258
tau0 = 0.47762
a = 3.59685
b = 8.68216

PYDirichmcmc =  MCMC(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, T, T0)
PYDirichDensity = dens(PYDirichmcmc, [8, 2], [-8, -6], 100)
PYDirichGroups = posteriorKn(PYDirichmcmc)

#PitmanYor - Stable 
sigma = 0.3253
sigma0 = 0.5281
theta =  0
theta0 = 1.7147
tau0 = 0.47762
a = 3.59685
b = 8.68216

PYStbmcmc =  MCMC(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, T, T0)
PYStbDensity = dens(PYStbmcmc , [8, 2], [-8, -6], 100)
PYStbGroups = posteriorKn(PYStbmcmc)



# Plots of densities

theme(:vibrant) 
p = histogram(y[1], normalized = true, linealpha = 0.1, bins =-8:0.7:8, color = :rosybrown3, fillalpha=0.2, label = nothing,
size=(800,500), foreground_color_legend = nothing, foreground_color_border=:gray68, framestyle = :box, foreground_color_axis=:gray68)
plot!(p, dirichletDensity.yy[1],  dirichletDensity.ff[:,1], markershape = :star5, markersize= 2, color=:orchid2, label = "HDP")
plot!(p, stableDensity.yy[1],  stableDensity.ff[:,1], markershape = :circle,  markersize= 2, color=:deeppink1, label = "HSP", line=:dashdot)
plot!(p, pitmanYorDensity.yy[1], pitmanYorDensity.ff[:,1], markershape = :utriangle, markersize = 2, color = :orange, label = "HPYP", line = :dash)
plot!(p,  dirPitmanYorDensity.yy[1],  dirPitmanYorDensity.ff[:,1], markershape = :star6, markersize= 2,color=:midnightblue,  label = "HDPYP", line=:dot)
plot!(p, StbPitmanYorDensity.yy[1], StbPitmanYorDensity.ff[:,1], markershape = :diamond, markersize= 2, color=:slateblue3, label = "HSPYP", line = :dashdotdot)
plot!(p, PYDirichDensity.yy[1],  PYDirichDensity.ff[:,1], color=:magenta3, label = "HPYDP", line=:dashdot)
plot!(p, PYStbDensity.yy[1],  PYStbDensity.ff[:,1], label = "HPYSP", color=:turquoise, line=:dash)
p

savefig(p, "densy1.png")

q = histogram(y[2], normalized = true, linealpha = 0.1,  bins =-6:0.5:2, color = :rosybrown3, fillalpha=0.2,
size=(800,500), label = nothing, foreground_color_legend = nothing, foreground_color_border=:gray68, framestyle = :box, foreground_color_axis=:gray68)
plot!(q, dirichletDensity.yy[2],  dirichletDensity.ff[:,2], markershape = :star5, markersize= 2, color=:orchid2, label = "HDP")
plot!(q, stableDensity.yy[2],  stableDensity.ff[:,2], markershape = :circle,  markersize= 2, color=:deeppink1, label = "HSP", line=:dashdot)
plot!(q, pitmanYorDensity.yy[2], pitmanYorDensity.ff[:,2], markershape = :utriangle, markersize = 2, color = :orange, label = "HPYP", line = :dash)
plot!(q,  dirPitmanYorDensity.yy[2],  dirPitmanYorDensity.ff[:,2], markershape = :star6, markersize= 2,color=:midnightblue,  label = "HDPYP", line=:dot)
plot!(q, StbPitmanYorDensity.yy[2], StbPitmanYorDensity.ff[:,2], markershape = :diamond, markersize= 2, color=:slateblue3, label = "HSPYP", line = :dashdotdot)
plot!(q, PYDirichDensity.yy[2],  PYDirichDensity.ff[:,2], markershape = :dtriangle, markersize= 2, color=:magenta3, label = "HPYDP", line=:dashdot)
plot!(q, PYStbDensity.yy[2],  PYStbDensity.ff[:,2], markershape = :star7, markersize= 2, label = "HPYSP", color=:turquoise, line=:dash)
q


savefig(q, "densy2.png")


# Plots of posterior distribution of K1
k1 = plot(dirichletGroups.unqi[1],  dirichletGroups.probs[1], markershape = :star5, markersize= 3, linewidth=2,
color=:orchid2, label = "HDP", foreground_color_legend = nothing, size=(800,500),  foreground_color_border=:gray68, framestyle = :box, foreground_color_axis=:gray68)
plot!(k1, stableGroups.unqi[1],  stableGroups.probs[1], markershape = :circle,  markersize= 3, color=:deeppink1,
label = "HSP", line=:dashdot, linewidth=2)
plot!(k1, pitmanYorGroups.unqi[1], pitmanYorGroups.probs[1], markershape = :utriangle, markersize = 3, color = :orange,
label = "HPYP", line = :dash, linewidth=2)
plot!(k1,  dirPitmanYorGroups.unqi[1],  dirPitmanYorGroups.probs[1], markershape = :star6, markersize= 3,color=:midnightblue,
label = "HDPYP", line=:dot, linewidth=2)
plot!(k1, StbPitmanYorGroups.unqi[1], StbPitmanYorGroups.probs[1], markershape = :diamond, markersize= 3, color=:slateblue3,
label = "HSPYP", line = :dashdotdot, linewidth=2)
plot!(k1, PYDirichGroups.unqi[1],  PYDirichGroups.probs[1], markershape = :dtriangle, markersize= 3, color=:magenta3,
label = "HPYDP", line=:dashdot, linewidth=2)
plot!(k1, PYStbGroups.unqi[1],  PYStbGroups.probs[1], markershape = :star7, markersize= 3, label = "HPYSP", color=:turquoise,
line=:dash, linewidth=2)
k1
savefig(k1, "posteriork1.png")


# # # # # k1 PLOTS K2 # # # # #
k2 = plot(dirichletGroups.unqi[2],  dirichletGroups.probs[2], markershape = :star5, markersize= 3, linewidth=2,
color=:orchid2, label = "HDP", foreground_color_legend = nothing, size=(800,500), foreground_color_border=:gray68, framestyle = :box, foreground_color_axis=:gray68)
plot!(k2, stableGroups.unqi[2],  stableGroups.probs[2], markershape = :circle,  markersize= 3, color=:deeppink1,
label = "HSP", line=:dashdot, linewidth=2)
plot!(k2, pitmanYorGroups.unqi[1], pitmanYorGroups.probs[1], markershape = :utriangle, markersize = 3, color = :orange,
label = "HPYP", line = :dash, linewidth=2)
plot!(k2,  dirPitmanYorGroups.unqi[2],  dirPitmanYorGroups.probs[2], markershape = :star6, markersize= 3,color=:midnightblue,
label = "HDPYP", line=:dot, linewidth=2)
plot!(k2, StbPitmanYorGroups.unqi[2], StbPitmanYorGroups.probs[2], markershape = :diamond, markersize= 3, color=:slateblue3, label = "HSPYP",
line = :dashdotdot, linewidth=2)
plot!(k2, PYDirichGroups.unqi[2],  PYDirichGroups.probs[2], markershape = :dtriangle, markersize= 3, color=:magenta3,
 label = "HPYDP", line=:dashdot, linewidth=2)
plot!(k2, PYStbGroups.unqi[2],  PYStbGroups.probs[2], markershape = :star7, markersize= 3, label = "HPYSP",
color=:turquoise, line=:dash, linewidth=2)
k2
savefig(k2, "posteriork2.png")


# Posterior of K0
k0 = plot(dirichletGroups.unq0,  dirichletGroups.p0, markershape = :star5, markersize= 4, linewidth=2,
color=:orchid2, label = "HDP", foreground_color_legend = nothing, xticks=(1:15), size=(800,500),foreground_color_border=:gray68, framestyle = :box, foreground_color_axis=:gray68)
plot!(k0, stableGroups.unq0,  stableGroups.p0, markershape = :circle,  markersize= 4, color=:deeppink1,
label = "HSP", line=:dashdot, linewidth=2)
plot!(k0, pitmanYorGroups.unq0, pitmanYorGroups.p0, markershape = :utriangle, markersize = 4, color = :orange,
label = "HPYP", line = :dash, linewidth=2)
plot!(k0,  dirPitmanYorGroups.unq0,  dirPitmanYorGroups.p0, markershape = :star6, markersize= 4,color=:midnightblue,
label = "HDPYP", line=:dot, linewidth=2)
plot!(k0, StbPitmanYorGroups.unq0, StbPitmanYorGroups.p0, markershape = :diamond, markersize= 4, color=:slateblue3,
label = "HSPYP", line = :dashdotdot, linewidth=2)
plot!(k0, PYDirichGroups.unq0,  PYDirichGroups.p0, markershape = :dtriangle, markersize= 4, color=:magenta3, label = "HPYDP", line=:dashdot)
plot!(k0, PYStbGroups.unq0,  PYStbGroups.p0, markershape = :star7, markersize= 4, label = "HPYSP", color=:turquoise,
line=:dash, linewidth=2)
k0
savefig(k0, "k0.png")



# Dataframe with goodness of fit
DirichletLPML = lpml(Dirichletmcmc, y)
stableLPML = lpml(Stablemcmc, y)
pitmanYorLPML = lpml(pitmanYormcmc, y)
PYDirichLPML = lpml(PYDirichmcmc, y)
PYStbLPML = lpml(PYStbmcmc, y)
StbPitmanYorLPML = lpml(StbPitmanYormcmc, y)
dirPitmanYorLPML = lpml(dirPitmanYormcmc, y)

logMarginalLikelihoods = [DirichletLPML.LPML, stableLPML.LPML, pitmanYorLPML.LPML,
PYDirichLPML.LPML, PYStbLPML.LPML, StbPitmanYorLPML.LPML, dirPitmanYorLPML.LPML]

groupOnelogMarginalLikelihoods = [DirichletLPML.LPMLi[1], stableLPML.LPMLi[1], pitmanYorLPML.LPMLi[1],
PYDirichLPML.LPMLi[1], PYStbLPML.LPMLi[1], StbPitmanYorLPML.LPMLi[1], dirPitmanYorLPML.LPMLi[1]]

groupTwologMarginalLikelihoods = [DirichletLPML.LPMLi[2], stableLPML.LPMLi[2], pitmanYorLPML.LPMLi[2],
PYDirichLPML.LPMLi[2], PYStbLPML.LPMLi[2], StbPitmanYorLPML.LPMLi[2], dirPitmanYorLPML.LPMLi[2]]
df = DataFrame(GlobalLPML = logMarginalLikelihoods,
               One = groupOnelogMarginalLikelihoods,
               Two = groupTwologMarginalLikelihoods)
CSV.write("Scores.csv", df)


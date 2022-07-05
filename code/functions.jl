using Random
using Distributions
using Statistics
using SpecialFunctions
using StatsPlots

#Random.seed!(1234)

# One iteration of CRF sampler
function iteration(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, c, d, l_i, l_d, q_bullet, Kn)

    #Sample label for Y_ij after removing the observation from the sample
     for i in 1:I
        for j in 1:n[i]
            # Remove Y_ij from current seat
            aux_c = c[i][j]
            aux_d = d[i][aux_c]
            q_bullet[i][aux_c] -= 1

            emptyTable = (q_bullet[i][aux_c] == 0)
            if emptyTable
                l_d[aux_d] -= 1
            end
            emptyDish = (l_d[aux_d] == 0)


            # Predictive distribution weights
            # Outer layer
            # if (sigma0==0 && sigma==0)
            #     w =  convert(Array{Float64,1}, l_d.-sigma0)
            # else
            #     w = l_d.-sigma0
            # end

            w =  convert(Array{Float64,1}, l_d.-sigma0)


            #w = l_d.-sigma0
            if emptyDish w[aux_d] = 0 end
            #prepend!(w, theta0 + sigma0*Kn)
            #w = (theta + l_i[i]*sigma).*(w./sum(w))

            !emptyDish ?  prepend!(w, theta0 + sigma0*Kn) :  prepend!(w, theta0 + sigma0*(Kn-1))
            !emptyTable ?  w = (theta + l_i[i]*sigma).*(w./sum(w)) : w = (theta + (l_i[i]-1)*sigma).*(w./sum(w))

            # Inner layer
            wTilde = q_bullet[i].-sigma
            if emptyTable wTilde[aux_c] = 0 end

            w = vcat(w, wTilde)

            # Likelihood on each component
            f_k = ones(Kn+1)

            scale_one = b/a*(tau0+1)/tau0
            # First position is new component
            f_k[1] = log(pdf(TDist(2*a), (y[i][j]-mu0)/sqrt(scale_one))/sqrt(scale_one))
            #pdf(TDist(2*a), (y[i][j]-mu0)/sqrt(scale_one))/sqrt(scale_one)


            for k = 1:Kn
                #S_k \ {(i,j)}
                Sk = []

                for i_aux = 1:I
                    index = findall(x->x == k, d[i_aux][c[i_aux]])
                    if i_aux == i
                        index = unique(setdiff(index, j))
                    end
                    append!(Sk, y[i_aux][index])
                end
                if !isempty(Sk)
                    sizeSk = length(Sk)
                    sumSk = sum(Sk)
                    meanSk = sumSk/sizeSk

                    tau0_aux = tau0 + sizeSk
                    mu0_aux = (tau0*mu0 + sumSk)/tau0_aux
                    a_aux = a + sizeSk/2
                    b_aux  = b + 0.5*(sum((Sk .- meanSk).^2) + tau0*sizeSk*(meanSk - mu0)^2/tau0_aux)
                    scale_aux = b_aux/a_aux*(tau0_aux + 1)/tau0_aux

                    f_k[k + 1] = log(pdf(TDist(2*a_aux), (y[i][j]-mu0_aux)/sqrt(scale_aux))/sqrt(scale_aux))
                    #pdf(TDist(2*a_aux), (y[i][j]-mu0_aux)/sqrt(scale_aux))/sqrt(scale_aux)

                end
            end
            f_k = f_k[vcat(1:Kn+1, d[i].+1)]

            #f_k = f_k.*w
            f_k = exp.(f_k .- maximum(f_k)).*w
            f_k = f_k/sum(f_k)

            cumsum_f_k = cumsum(f_k)
            u = rand()
            cc = findfirst(x-> u <= x, cumsum_f_k)

            if cc == 1 #New cluster with new label

                if emptyTable #See if reuse labels
                    q_bullet[i][aux_c] = 1
                    if emptyDish
                        l_d[aux_d] = 1
                    else
                        Kn += 1
                        append!(l_d, 1)
                        d[i][aux_c] = Kn
                    end
                else #If not create new labels
                    Kn += 1
                    #append!(l_d, 1)
                    push!(l_d, 1)
                    l_i[i] += 1
                    c[i][j] = l_i[i]
                    #append!(d[i], Kn)
                    push!(d[i], Kn)
                    append!(q_bullet[i], 1)
                end
            elseif cc <= Kn+1 #New cluster but with old label
                cc -= 1
                l_d[cc] +=1
                if emptyTable
                    d[i][aux_c] = cc
                    q_bullet[i][aux_c] = 1
                    if emptyDish
                        Kn -= 1
                        for i_j = 1:I
                            d[i_j][d[i_j] .> aux_d].-= 1
                        end
                        deleteat!(l_d, aux_d)
                    end
                else
                    l_i[i] += 1
                    c[i][j] = l_i[i]
                    append!(d[i], cc)
                    append!(q_bullet[i], 1)
                end
            else #Old cluster with old label
                cc = cc - Kn - 1
                c[i][j] = cc
                q_bullet[i][cc] += 1

                if emptyTable # Delete unused clusters
                    deleteat!(q_bullet[i], aux_c)
                    c[i][c[i] .> aux_c] .-= 1
                    deleteat!(d[i], aux_c)
                    l_i[i] -= 1
                    if emptyDish
                        Kn -= 1
                        for i_j = 1:I
                            d[i_j][d[i_j] .> aux_d].-= 1
                        end
                        deleteat!(l_d, aux_d)
                    end
                end
            end
        end
    end


    #Sample the table label after removing all observations at that table from the list
    for i = 1:I
        num_tables = length(unique(c[i]))

        for t = 1:num_tables
            # Remove Y_ic from current table
            aux_d = d[i][t]
            l_d[aux_d] -= 1
            emptyDish = (l_d[aux_d] == 0)

            f_k = ones(Kn+1)

            #Select the tables serving dish k
            table_j = findall(x-> x == t, c[i])
            table = y[i][table_j]

            sizetable = length(table)
            sumtable = sum(table)
            meantable = sumtable/sizetable
            tau0_table = tau0 + sizetable
            mu0_table = (tau0*mu0 + sumtable)/tau0_table
            a_table = a + sizetable/2
            b_table = b + 0.5*(sum((table .- meantable).^2) + tau0*sizetable*(meantable - mu0)^2/tau0_table)
            # First position is new dish
            f_k[1] = loggamma(a_table) - loggamma(a) + 0.5*(log(tau0) - log(tau0_table)) - sizetable/2*(log(2*pi)) + a*log(b) - a_table*log(b_table)

            for k = 1:Kn
                # Sk \ {Y_ic}
                Sk = []
                for i_aux = 1:I
                    index = findall(x->x == k, d[i_aux][c[i_aux]])
                    if i_aux == i
                        index = unique(setdiff(index, table_j))
                    end
                    Sk = append!(Sk, y[i_aux][index])
                end

                if !isempty(Sk) #If not empty compute the likelihood
                    sizeSk = length(Sk)
                    sumSk = sum(Sk)
                    meanSk = sumSk/sizeSk
                    meanboth = (sumtable + sumSk)/(sizetable + sizeSk)

                    tau0_Sk = tau0 + sizeSk
                    tau0_aux = tau0_Sk + sizetable
                    mu0_Sk = (tau0*mu0 + sumSk)/tau0_Sk
                    mu0_aux = (tau0*mu0 + sumSk + sumtable)/tau0_aux
                    a_Sk = a + sizeSk/2
                    a_aux = a_Sk + sizetable/2
                    b_Sk = b + 0.5*(sum((Sk .- meanSk).^2) + tau0*sizeSk*(meanSk - mu0)^2/tau0_Sk)
                    b_aux = b + 0.5*(sum((vcat(table, Sk) .- meanboth).^2) + tau0*(sizetable + sizeSk)*(meanboth - mu0)^2/tau0_aux)

                    f_k[k+1] = loggamma(a_aux) - loggamma(a_Sk) + .5*(log(tau0_Sk) - log(tau0_aux)) - sizetable/2*(log(2*pi)) + a_Sk*log(b_Sk) - a_aux*log(b_aux);

                end
            end
            #Predictive distribution weights
            # THIS PREVENTS ERROR FROM TRYING TO CONVERT THETA TO INTEGER (as l_d is constructed as an integer array)
            #sigma0==0 && sigma==0 ? w =  convert(Array{Float64,1}, l_d.-sigma0) : w = l_d.-sigma0

            w =  convert(Array{Float64,1}, l_d.-sigma0)

            #w = l_d.-sigma0
            if emptyDish w[aux_d]=0 end
            !emptyDish ?  prepend!(w, theta0 + sigma0*Kn) :  prepend!(w, theta0 + sigma0*(Kn-1))
            #prepend!(w, theta0 + Kn*sigma0)

            f_k = exp.(f_k .- maximum(f_k)).*w
            f_k = f_k/sum(f_k)

            cumsum_f_k = cumsum(f_k)
            u = rand()

            d_new = findfirst(x-> u <= x, cumsum_f_k)

            if  d_new == 1 #
                if emptyDish
                    #d[i][aux_d]= 1 # creo que esto no va
                    l_d[aux_d] = 1
                else
                    Kn += 1
                    append!(l_d, 1)
                    d[i][t] = Kn
                end
            else
                h_k = d_new - 1

                l_d[h_k] += 1
                d[i][t] = h_k
                if emptyDish
                    Kn -= 1
                    for i_j = 1:I
                        d[i_j][d[i_j] .> aux_d].-= 1 # recorre indices
                    end
                    deleteat!(l_d, aux_d)
                end
            end
        end
    end

    # Posterior parameters
    Sigma2 = zeros(Kn)
    Mu = zeros(Kn)
    for k = 1:Kn
        dish_k = []
        for i = 1:I
            dish_k = vcat(dish_k, y[i][findall(x->x == k, d[i][c[i]])])
        end
        if !isempty(dish_k)
            size_k = length(dish_k)
            sum_k = sum(dish_k)
            mean_k = sum_k/size_k
            tau0_aux = tau0 + size_k
            Sigma2[k] = 1/rand(Gamma(a+size_k/2,1/(b+0.5*(sum((dish_k.-mean_k).^2) + tau0*size_k*(mean_k - mu0)^2/tau0_aux))))
            Mu[k] = rand(Normal((tau0*mu0+sum_k)/tau0_aux, sqrt(Sigma2[k]/tau0_aux)))
        end
    end

    return(c = c, d = d, l_i = l_i, l_d = l_d, q_bullet = q_bullet, Kn = Kn, Mu = Mu, Sigma2 = Sigma2)
end



# MCMC main algorithm
function MCMC(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, T, T0)
    I = length(y)
    n = [length(y[i]) for i in 1:I]

    # Initial configuration
    c = [ones(Int64, n[j]) for j in 1:I]
    d = [ones(Int64, 1) for j in 1:I]
    l_i = ones(Int64, I)
    l_d = [I]
    q_bullet =[ones(Int64, 1).*n[j] for j in 1:I]
    Kn = 1

    # Outputs - #YA FUNCIONAN
    c_out = [[[] for i in 1:I] for t in 1:T]
    q_bullet_out = [[[] for i in 1:I] for t in 1:T]
    d_out = [[[] for i in 1:I] for t in 1:T]
    l_i_out = [[] for i in 1:T]
    l_d_out = [[] for i in 1:T]
    Kn_out = zeros(Int64, T)

    # Output parameters
    Mu_out = [[] for i in 1:T]
    Sigma2_out = [[] for i in 1:T]

    if(T0>0)
        print("Burning-in \n")
        for t in 1:T0
            sim = iteration(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, c, d, l_i, l_d, q_bullet, Kn)
            c = sim.c
            d = sim.d
            l_i = sim.l_i
            l_d = sim.l_d
            q_bullet = sim.q_bullet
            Kn = sim.Kn
        end
    end
    print("Sampling \n")
    if (T>0)
        for t in 1:T
            sim = iteration(y, sigma, sigma0, theta, theta0, mu0, tau0, a, b, c, d, l_i, l_d, q_bullet, Kn)

            c = sim.c
            d = sim.d
            l_i = sim.l_i
            l_d = sim.l_d
            q_bullet = sim.q_bullet
            Kn = sim.Kn

            # Saving the outputs
            c_out[t] = sim.c
            d_out[t] = sim.d
            l_i_out[t] = sim.l_i #setindex!(l_i_out, sim.l_i, t)
            l_d_out[t] = sim.l_d #setindex!(l_d_out, sim.l_d, t)
            q_bullet_out[t] = sim.q_bullet
            Kn_out[t] = sim.Kn

            Mu_out[t] = sim.Mu
            Sigma2_out[t] = sim.Sigma2
            end
    end
    return (c = c_out, d = d_out, l_i = l_i_out, l_d = l_d_out, q_bullet = q_bullet_out, Kn = Kn_out, Mu = Mu_out, Sigma2 = Sigma2_out)
    #return c_out, d_out, l_i_out, l_d_out, q_bullet_out, Kn_out, Mu_out, Sigma2_out
end



# Gaussian Kernel
function kernel(y, μ, σ)
    return pdf(Normal(μ, σ), y)
end


# Density estimation
function dens(mcmc, ymax, ymin, sizegrid)
    I = length(mcmc.c[1])
    yy = [range(ymin[i], ymax[i], length = sizegrid) for i in 1:I]
    ff = zeros(sizegrid, I)
    T = length(mcmc.c) #iterations
    nn = [length(mcmc.c[1][i]) for i in 1:I] #n_is

    for i=1:I
        for m=1:T
            numTables = mcmc.l_i[m][i]  #number of groups on iteration m in subset i
            for c in 1:numTables
                index = mcmc.d[m][i][c]
                for l=1:sizegrid
                    ff[l, i] += mcmc.q_bullet[m][i][c]*kernel(yy[i][l], mcmc.Mu[m][index], sqrt(mcmc.Sigma2[m][index]))
                end
            end
        end
        ff[:,i] = ff[:,i]./(T.*nn[i])
    end
    return(yy = yy, ff = ff)
end


# Posterior of K_N and Kni's
function posteriorKn(mcmc)
    I = length(mcmc.c[1])

    # Global - K_0
    unq0 = sort(unique(mcmc.Kn))
    p0 = zeros(length(unq0))
    T = length(mcmc.Kn) #iterations
    for j in 1:length(unq0)
        p0[j] = count(x -> (x==unq0[j]), mcmc.Kn)/T
    end

    # Per group
    Ki = [[] for i in 1:I]
    for i in 1:I
        for m in 1:T
            append!(Ki[i], mcmc.l_i[m][i])
        end
    end
    unqi = [[] for i in 1:I]
    for i in 1:I
        unqi[i] = sort(unique(Ki[i]))
    end
    probs = [[] for i in 1:I]
    for i in 1:I
        for j in 1:length(unqi[i])
            append!(probs[i], count(x -> (x==unqi[i][j]), Ki[i])/T)
        end
    end

    return(unq0 = unq0, p0 = p0, unqi = unqi, probs = probs)
end


# Log-pseudo marginal likelihood
function lpml(mcmc, y)
    I = length(y)
    T = length(mcmc.Kn)
    n = [length(y[i]) for i in 1:I]

    CPO_i = [[] for i in 1:I]
    CPO = zeros(sum(n))

    for m in 1:T
        for i in 1:I
            index = mcmc.d[m][i][mcmc.c[m][i]]
            ct = pdf.(Normal.(mcmc.Mu[m][index], sqrt.(mcmc.Sigma2[m][index])), y[i])
            CPO_i[i] = 1 ./ct
        end
        CPO = CPO + vcat(CPO_i...)
    end
    LPML_i =[sum(log.(T./ CPO_i[i])) for i in 1:I]
    LPML = sum(log.(T ./ CPO ))
    #LPML_i =[sum(log.(T ./ CPO_i[i])) for i in 1:I]
    #LPML = sum(log.(T ./ CPO ))
    return(LPML = LPML, LPMLi = LPML_i)
end


# Deviance
# function deviance(mcmc, y)
#     I = length(y)
#     T = length(mcmc.Kn)
#     n = [length(y[i]) for i in 1:I]
#
#     dev_i = [[] for i in 1:I]
#     dev = zeros(sum(n))
#
#     for m in 1:T
#         for i in 1:I
#             index = mcmc.d[m][i][mcmc.c[m][i]]
#             ct = pdf.(Normal.(mcmc.Mu[m][index], sqrt.(mcmc.Sigma2[m][index])), y[i])
#             dev_i[i] = mcmc.l_d[m][(index)].*ct/n[i]
#         end
#         dev .= dev + vcat(dev_i...)
#     end
#     deviance = -2*sum(log.(dev))
#     return deviance
# end

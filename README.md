# Mixtures for partially exchangeable data
## HNRMIs as prior distributions of mixture models
Hierarchical normalized random measures with independent increments (HNRMIs) can be used as prior distributions for the components of a mixture model.  Assume that 

$$
\begin{align*}
(\mathrm{y}_{i,j})\_{i=1}^{n_i} \text{ for }i=1,\ldots, m
\end{align*}
$$

are presumably partially exchangeable. Given an unobserved parameter $\mathrm{x}\_{i,j}$, the data generating mechanism is

$$ 
  \begin{equation*}
\begin{split}
     \mathrm{y}\_{i,j} \, | \, \mathrm{x}\_{i,j} &\stackrel{\text{ind}}{\sim} \mathcal{K}(\, \cdot \, | \, \mathrm{x}\_{i,j}) \quad j=1,\ldots, n_i, \, i=1,\ldots, m \\
    \mathrm{x}\_{i,j} \, | \, \tilde{p}_i &\sim \text{NRMI}(\rho, \theta, \tilde{p}_0) \\
    \tilde{p}_0 &\sim \text{NRMI}(\rho_0, \theta_0, P_0),
\end{split}
\end{equation*} 
$$

where $\mathcal{K}$ is a suitable kernel density. 
By using the chinese restaurant franchise sampler, described in X Y W, we can build a Gibbs sampler that subsequently samples the 'local' mixture component of each point $(\mathrm{y}_{i,j})\_{i=1}^{n_i}$ of each of the $m$ groups, and then 'global' component overall. We can think of this as if we were sampling 


## Toy example
In the above model, assume that $\mathcal{K}$ is a Gaussian kernel and that, to achieve conjugacy, the base measure $P_0$ is a normal-gamma distribution. We simulated 200 data points from the mixtures

$$
\begin{align*}
\mathrm{y}_{1,j} &\sim 0.6\mathrm{N}(-4, 1) + 0.2 \mathrm{N}(0, 0.5) + 0.2\mathrm{N}(3.5, 1.25)\quad j=1,\ldots, 100 \\
    \mathrm{y}_{2,j} &\sim 0.7\mathrm{N}(0, 0.5) + 0.3 \mathrm{N}(-3.5, 1) \quad j=1,\ldots, 100 
\end{align*}
$$

The common component is shown in blue. The goal of the sampler is to fit density estimations on both groups whilst recognizing that the component shown in blue is the same across the two groups. This means that we 
The fitted densities are shown in the next picture. The acronyms mean respectively: hierarchical Dirichlet process (HDP), hierarchical stable process (HSP), hierarchical Pitman-Yor process (HPYP), 

# Model, Inversion and Sampling Objectives

[Project overview](../README.md) · [Results](results.md) · [Reproduction](reproduction.md)

## Physical setting

The geometry follows the heterogeneous 18-row × 14-column MT3DMS Problem 9 application example, with 100 m base cells and a 10 m aquifer thickness. Background conductivity is 1.474 × 10⁻⁴ m/s; the low-conductivity lens is 1.474 × 10⁻⁷ m/s. The lens occupies base rows 5–7 and columns 1–7, using zero-based indexing. Injection occurs in base cell (3, 6); extraction occurs in (10, 6). Boundary heads are prescribed on the top and bottom strips, with no flow through the sides.

The custom model uses metres and days. Hydraulic conductivities and well flows are converted accordingly in [`Config`](../physics.py). Main-study refinement divides each base cell into 4 × 4 cells, giving a 72 × 56 computational grid. The 100 m source and observation footprints remain fixed under refinement. Measurements are footprint averages rather than values from whichever refined cell happens to contain a point.

Steady saturated flow uses harmonic face conductances. For an oriented face from cell $i$ to cell $j$,

$$
F_{ij}=G_{ij}(h_i-h_j),
\qquad \sum_j F_{ij}=Q_i+Q_i^{\mathrm{boundary}}.
$$

The transport semidiscretization is

$$
M\frac{d c}{dt}+L c=s u(t).
$$

$M$ contains pore volumes; $L$ combines first-order upwind advection, diagonal hydrodynamic dispersion, extraction and external outflow. Longitudinal and transverse dispersivities are 20 m and 4 m. Concentration entering through the external boundary is zero. Backward Euler advances the model; each step is split exactly at observation times and release transitions. Internal advective and dispersive face fluxes cancel pairwise in the mass balance.

This is a simplified independent implementation. It is not the TVD/full-dispersion solver used in the original public example. In particular, cross-dispersion terms are omitted. [`benchmark_mf6.py`](../benchmark_mf6.py) compares the independent implementations only after matching their equations: steady flow, upstream advection, zero dispersion and 1-day steps.

## Uncertain source and conductivity

The uncertain vector is

$$
\theta=(\eta_b,\eta_l,a_1,a_2,a_3).
$$

The first two entries are natural-log multipliers of background and lens conductivities, with independent normal priors of standard deviations 0.35 and 0.70. Each source amplitude has an independent lognormal prior with median 57.87 mg/L and log standard deviation 0.45. The three amplitudes apply during days 0–120, 120–240 and 240–360 at the same known injection footprint.

Transport is linear in amplitudes for fixed conductivity. The solver therefore computes three unit-amplitude basis responses for each conductivity pair. Concentrations for all amplitude combinations are obtained by linear combination. This reduces the main response library to nine conductivity solves while representing all $3^2 5^3=1125$ weighted parameter states.

The quadrature rule uses three Gauss–Hermite nodes on each conductivity axis and five on each source axis. Source axes form a full independent tensor product. The weighted ensemble approximates a continuous prior; exact summation on this finite ensemble does not imply accurate integration of a narrow continuous posterior.

## Observations and posterior

Initial synthetic observations at day 180 comprise concentration in base cells (4, 6), (8, 6), (10, 6), and head in cells (6, 3), (9, 10). Errors are independent Gaussian with standard deviations 0.08 mg/L for concentration and 1.5 m for head. Candidate observations are day-365 concentrations at 24 virtual monitoring footprints. The target is day-545 concentration in base cell (10, 6).

For prior weights $\pi_k$ and observation predictions $g_k$, the normalized posterior is

$$
w_k(y)=\frac{\pi_k\exp[-\tfrac12(y-g_k)^T\Sigma^{-1}(y-g_k)]}
{\sum_l\pi_l\exp[-\tfrac12(y-g_l)^T\Sigma^{-1}(y-g_l)]}.
$$

The implementation evaluates normalization in log space with `logsumexp`. Central 95% intervals use generalized-inverse discrete CDF endpoints: the first sorted target value whose cumulative weight reaches 0.025 or 0.975. They do not interpolate between ensemble nodes. Effective support size is $1/\sum_k w_k^2$.

A causal check follows from source timing. The third release begins at day 240, after the initial observations. Its response at day 180 must be identically zero. Because source priors are independent, day-180 observations must also leave its entire marginal prior unchanged. [`check_inference.py`](../check_inference.py) verifies both properties across all archived main cases and checks composition of independent likelihood factors.

## Batch sampling

The action is whether to strengthen monitoring. Let $q(\theta)$ be receptor concentration and $p=P(q>0.6\mid y)$. With false-negative cost 9 and false-positive cost 1, the optimal posterior risk is

$$
r(p)=\min\lbrace 9p,1-p\rbrace.
$$

Strengthening is selected when $p>0.1$. Equality uses the non-strengthening action in the implementation. The concentration threshold and costs are experimental action settings.

A selected batch $S$ is evaluated before observing its new data. Its decision objective is

$$
R(S)=\mathbb{E}_{Y_S\mid y}[r(P(q>0.6\mid y,Y_S))].
$$

Four criteria are compared with the same four-location budget:

| Method | Selection criterion |
|---|---|
| Spatial coverage | Greedily maximize distance to the initial concentration footprints and already selected locations |
| Parameter information | Maximize expected KL divergence of the joint five-parameter posterior from the current posterior |
| Predictive variance | Minimize expected posterior variance of receptor concentration |
| Decision loss | Minimize expected optimal threshold-decision loss |

For the three model-based criteria, greedy construction adds one location at a time, rescoring the joint prospective observations of the entire provisional batch. The four selected observations are then collected simultaneously. The construction order is not a sequence of adaptive observations.

Selection uses 256 simulated predictive draws with common random numbers across candidate sets. Posterior summation for each draw is exact on the finite support. Evaluation uses an independent set of 4,096 conditional predictive draws, shared across methods within a case. Its inner calculation is a weighted sum, not a plug-in empirical estimate of exceedance probability from a second inner Monte Carlo sample.

An additional illustrative observed update in each saved case generates one actual synthetic measurement vector. This differs from averaging over prospective data; fields named `illustrative_after` describe that one update.

## Why multiple validation layers matter

Conservation, positivity and analytical checks test implementation properties. Independent conditional draws test whether the design overfits its own sampled objective. Finite-support held-out cases test the stated finite model. None resolves quadrature collapse when truth lies between its parameter nodes.

The supplementary continuous-prior experiments draw parameters directly from the continuous distributions and solve the PDE for those parameters. The mesh/time mismatch group also uses a finer truth solver. Both expose weaknesses that finite-model predictive evaluation can miss. The study therefore treats the finite model as a computational experiment, with uncertainty calibration and numerical resolution as unresolved requirements for practical use.

## References

- MODFLOW 6 Examples, [MT3DMS Problem 9: Two-Dimensional Application Example](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwt-mt3dms-p09.html). The geometry and physical parameter values derive from this public benchmark. The archived source reference is [commit ec5ac613](https://github.com/MODFLOW-ORG/modflow6-examples/blob/ec5ac6134d3d25684fbdd5f204864258610f2330/scripts/ex-gwt-mt3dms-p09.py). This repository adds uncertain releases, Bayesian inversion, monitoring candidates and action costs.
- Langevin, C. D., Provost, A. M., Panday, S., and Hughes, J. D. (2022). [Documentation for the MODFLOW 6 Groundwater Transport Model](https://pubs.usgs.gov/publication/tm6A61). USGS Techniques and Methods 6-A61.
- Attia, A., Alexanderian, A., and Saibaba, A. K. (2018). [Goal-Oriented Optimal Design of Experiments for Large-Scale Bayesian Linear Inverse Problems](https://arxiv.org/abs/1802.06517). *Inverse Problems*. DOI: 10.1088/1361-6420/aad210.
- Huan, X., Jagalur, J., and Marzouk, Y. (2024). [Optimal experimental design: Formulations and computations](https://arxiv.org/abs/2407.16212). *Acta Numerica*. DOI: 10.1017/S0962492924000023.
- Lykkegaard, M. B., and Dodwell, T. J. (2022). [Where to Drill Next? A Dual-Weighted Approach to Adaptive Optimal Design of Groundwater Surveys](https://arxiv.org/abs/2111.07670). *Advances in Water Resources*. DOI: 10.1016/j.advwatres.2022.104219.

Bayesian inversion, goal-oriented design and decision-theoretic experimental design are established methods. This repository investigates their comparison within a specific reproducible synthetic setting.

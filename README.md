# Bayesian Hierarchical Skew-t Modeling and Copula Portfolio Risk

A Bayesian workflow for modeling equity log returns that captures the two
features a Gaussian model misses — **heavy tails** and **skewness** — and then
stitches the asset-level fits into a joint portfolio distribution with
**copulas** to study tail risk (Value-at-Risk, skewness, kurtosis).

The marginal returns are fit with a **Skewed Generalized t (SGT)** distribution
via a **Bayesian hierarchical model** sampled by MCMC (Gibbs sampling in JAGS);
the dependence structure is modeled by comparing five copulas and validating
them out of sample.

> Course/research project, NC State University. Advisor: Dr. Sujit Ghosh.

## Motivation

Standard tools (Black-Scholes, the Sharpe ratio) assume normally distributed log
returns. Real return series violate this: they have excess kurtosis (extreme
moves are more frequent than Normal predicts), asymmetry (downside moves are
often sharper than upside), and volatility clustering. Because risk measures
like VaR and Expected Shortfall are tail-sensitive, a Normal model
systematically understates the probability of large losses. The SGT marginal
plus a tail-aware copula is built to correct this.

## Methodology

**1. Exploratory analysis and goodness-of-fit.** For ten equities (MSFT, JNJ,
PG, PEP, WMT, TSLA, NVDA, AMD, AMZN, META), compare a Normal fit against a
skew-t fit using the Kolmogorov-Smirnov and Anderson-Darling tests (the AD test
is weighted toward the tails, so it is the more demanding of the two here).

**2. Bayesian hierarchical skew-t model.** Skewness is injected through a latent
half-normal variable. For stock `j` and observation `i`:

```
y[i,j]  ~ Normal(mu[j] + alpha[j] * z[i,j], tau[j])
z[i,j]  ~ HalfNormal                      # latent skewness driver, z > 0
mu[j]   ~ Normal(mu0, tau0)               # location
tau[j]  ~ Gamma(a0, b0),  sigma[j] = tau[j]^(-1/2)   # precision / scale
alpha[j]~ Normal(0, 1)                    # skewness
nu      ~ Gamma(2, 0.1)                   # tail heaviness (shared)
mu0 ~ Normal(0,1);  tau0 ~ Gamma(1e-4, 0.05);  a0, b0 ~ Gamma(0.1, 0.5)
```

Because `z` is constrained positive, the term `alpha[j] * z[i,j]` shifts the
mean asymmetrically — larger `|alpha[j]|` gives more skew. The hierarchy lets
location, scale, and skewness vary by stock while sharing hyperpriors, which is
suited to the wide spread of volatilities across the universe. Sampling uses
three overdispersed chains (30k adapt, 50k burn-in, 150k draws, thinned by 2)
so the between-chain Gelman-Rubin diagnostic is meaningful.

**3. Copula dependence.** Marginals are mapped to uniforms (empirical CDF for
the standard copulas; the fitted skew-t CDF for the skew-t copula) and five
copulas are compared — Gaussian, t, Clayton (lower-tail), Gumbel (upper-tail),
and skew-t — by AIC in sample and by log-likelihood out of sample.

**4. Monte Carlo portfolio simulation.** The best copula is sampled (10,000
draws), pushed back through the marginal quantile functions, and aggregated into
an equal-weight portfolio to estimate the return distribution and its tail
statistics.

## Data

Daily adjusted-close prices for the ten tickers over roughly 500 trading days
(about two years ending late 2024), converted to log returns. The price file is
not committed; `fetch_data.R` regenerates `data/stock_data.csv` from Yahoo
Finance. Yahoo back-adjusts history, so a fresh pull may differ slightly from
the figures below.

## Results

**The Normal distribution is rejected; the skew-t is not.** Under the Normal
fit, several stocks fail badly — META at the extreme (KS p ≈ 2e-5, AD p ≈ 1e-5),
along with WMT and JNJ. Under the skew-t fit, every stock clears the 5% bar on
both tests (all p > 0.8), e.g. META rises to KS 0.855 / AD 0.871.

| Fit       | Stocks passing KS & AD at 5% | Typical p-value |
|-----------|------------------------------|-----------------|
| Normal    | a minority (e.g. MSFT, PEP)  | many below 0.05 |
| Skew-t    | all 10                       | > 0.8           |

**MCMC converged.** For the 3-asset portfolio (AMZN, NVDA, PG), all
Gelman-Rubin potential scale reduction factors are ≈ 1.00-1.01. Effective sample
sizes range from ~1,565 to ~20,940 — the wide spread (and uneven mixing for a
few `alpha`/`mu` parameters) points to parameter correlation in the centered
parameterization, discussed in Limitations.

**Portfolio tail risk** (posterior summary of the equal-weight portfolio):

| Mean      | Std. dev. | Skewness | Kurtosis | 95% VaR |
|-----------|-----------|----------|----------|---------|
| -0.00168  | 0.0003    | 0.2194   | 10.53    | 0.0291  |

The kurtosis of ~10.5 confirms heavy tails that a Gaussian portfolio model would
miss entirely.

**The skew-t copula wins out of sample.** Lowest AIC and highest test
log-likelihood of the five:

| Copula   | AIC      | Test log-likelihood |
|----------|----------|---------------------|
| Gaussian | -60.13   | 45.98               |
| t        | -81.85   | -1573.58            |
| Clayton  | -30.04   | 4.38                |
| Gumbel   | -42.15   | 16.48               |
| **Skew-t** | **-1444.48** | **69.80**       |

## Limitations

- **The hierarchical model approximates the skew-t.** Skewness enters through a
  half-normal latent mixture, and the degrees-of-freedom parameter `nu` is given
  a prior but does not enter the likelihood directly, so tail-heaviness is
  captured only indirectly. A closed-form, fully hierarchical SGT — e.g. the
  stochastic representation in Lian, Rong & Cheng, *"On a Novel Skewed
  Generalized t Distribution"* (combining a skewed generalized normal with a
  gamma) — would be more faithful.
- **Parameter correlation.** Correlated `alpha`/`mu` inflate autocorrelation and
  produce the uneven effective sample sizes above. A non-centered
  reparameterization, and gradient-based samplers (HMC / NUTS), would mix better.
- **Out-of-sample design.** Validation trains each copula on the first 20% of
  the sample and scores the remaining 80% (set by `train_proportion = 0.2`).
  This is a held-out split, not a rolling/walk-forward backtest, and the
  t-copula's large negative test log-likelihood is an anomaly worth
  investigating before relying on the ranking.
- **Single window, equal weights.** One ~500-day sample and a fixed equal-weight
  portfolio; no rebalancing and no transaction costs.

## Possible extensions

- Non-centered reparameterization + HMC/NUTS (e.g. via Stan) to fix the mixing.
- The closed-form hierarchical SGT above for a faithful (not approximate) model.
- Walk-forward copula validation with a conventional train-majority split.
- Regularizing priors (Lasso/Ridge-type) to shrink parameters.

## Repository contents

```
bayesian_skew_t_copula.R   # full pipeline: EDA -> Bayesian MCMC -> copulas -> simulation
fetch_data.R               # regenerate data/stock_data.csv from Yahoo Finance
report.pdf                 # full write-up with all figures (histograms, QQ, traces, densities)
slides.pdf                 # presentation summary
data/stock_data.csv        # prices (regenerate with fetch_data.R; not committed)
README.md
```

## Running

JAGS must be installed system-wide (https://mcmc-jags.sourceforge.io/) before
the R `rjags` package can load.

```bash
Rscript fetch_data.R              # writes data/stock_data.csv
Rscript bayesian_skew_t_copula.R  # runs the full analysis
```

The MCMC stage (3 chains, 150k draws) is the slow part — expect several minutes.
Outputs `stock_distribution_plots.pdf` and `mcmc_diagnostics_skewt.pdf`, and
prints the goodness-of-fit, Bayesian, and copula tables to the console.

## Tools

R: `rjags` (Gibbs sampling / JAGS), `sn` (skew-t density/quantile/CDF),
`copula` (Gaussian, t, Clayton, Gumbel fitting & simulation), `fitdistrplus`
and `goftest` (ML fitting, KS/AD tests), `coda` (convergence diagnostics),
`moments`, `mvtnorm`, `MASS`.

## Credits

Author: Abhiram Vadlamani. Advised by Dr. Sujit Ghosh (NC State University).
The closed-form SGT direction noted above follows Lian, Rong & Cheng.

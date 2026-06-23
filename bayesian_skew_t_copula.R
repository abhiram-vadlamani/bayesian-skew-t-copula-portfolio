# =============================================================================
# Bayesian Hierarchical Modeling of Log Returns and Portfolio Risk Analysis
# -----------------------------------------------------------------------------
# Author : Abhiram Vadlamani
# Advisor: Dr. Sujit Ghosh (NC State University)
#
# Pipeline:
#   1. Exploratory data analysis + goodness-of-fit (Normal vs. Skew-t) for 10
#      equities, using Kolmogorov-Smirnov and Anderson-Darling tests.
#   2. Bayesian hierarchical Skew-t model fit by MCMC (Gibbs sampling via JAGS)
#      for a 3-asset portfolio (AMZN, NVDA, PG), with convergence diagnostics.
#   3. Copula dependence modeling (Gaussian, t, Clayton, Gumbel, Skew-t) with
#      in-sample fitting and out-of-sample log-likelihood validation.
#   4. Monte Carlo simulation of the joint portfolio return distribution and
#      tail-risk statistics (VaR, skewness, kurtosis).
#
# Reproducibility:
#   - Expects a CSV at `data_path` with a Date column followed by one price
#     column per ticker. Run fetch_data.R to regenerate it from Yahoo Finance.
#   - JAGS must be installed system-wide (https://mcmc-jags.sourceforge.io/)
#     for the rjags interface to load.
#
# Note on the model: the Skew-t likelihood is approximated by a half-normal
# latent-variable mixture (see Section 5). The degrees-of-freedom parameter nu
# is assigned a prior but does not enter the likelihood directly, so it captures
# tail-heaviness only indirectly. This is a deliberate simplification; see the
# README "Limitations" section for the exact, closed-form alternative.
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Setup
# -----------------------------------------------------------------------------
required_packages <- c("fitdistrplus", "sn", "actuar", "mvtnorm", "copula",
                       "readr", "MASS", "moments", "rjags", "goftest")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)
invisible(lapply(required_packages, library, character.only = TRUE))

# Reproducibility: portfolio stocks and the data file location.
data_path        <- "data/stock_data.csv"
selected_stocks  <- c("AMZN", "NVDA", "PG")  # 3-asset portfolio for Parts 2-4

# -----------------------------------------------------------------------------
# 1. Helpers
# -----------------------------------------------------------------------------

# Daily log returns from a price series.
calculate_log_returns <- function(stock_prices) {
  return(diff(log(stock_prices)))
}

# -----------------------------------------------------------------------------
# 2. Marginal distribution fitting
# -----------------------------------------------------------------------------

# Fit a skew-t distribution by maximum likelihood, run KS + AD goodness-of-fit
# tests, and fall back to a Normal fit if the optimizer fails to converge.
fit_skew_t_distribution <- function(log_returns, stock_name) {
  tryCatch({
    st_fit <- fitdist(log_returns, "st",
                      start = list(xi    = mean(log_returns),
                                   omega = sd(log_returns),
                                   alpha = 0,
                                   nu    = 10),
                      lower = c(-Inf, 0.001, -Inf, 2.1),  # omega > 0, nu > 2
                      upper = c(Inf, Inf, Inf, 50))

    # Safely extract and bound parameters.
    params <- st_fit$estimate
    xi    <- as.numeric(params["xi"])
    omega <- max(as.numeric(params["omega"]), 0.001)
    alpha <- as.numeric(params["alpha"])
    nu    <- min(as.numeric(params["nu"]), 50)

    ks_test <- ks.test(log_returns, "pst",
                       xi = xi, omega = omega, alpha = alpha, nu = nu)
    ad_test <- ad.test(log_returns,
                       function(x) pst(x, xi = xi, omega = omega,
                                       alpha = alpha, nu = nu))

    list(
      fit        = st_fit,
      ks_test    = ks_test,
      ad_test    = ad_test,
      parameters = c(xi = xi, omega = omega, alpha = alpha, nu = nu)
    )
  }, error = function(e) {
    warning(paste("Error fitting skew-t distribution for", stock_name, ":", e$message))

    # Fallback to Normal if skew-t estimation fails.
    norm_fit <- fitdist(log_returns, "norm")
    list(
      fit     = norm_fit,
      ks_test = ks.test(log_returns, "pnorm",
                        mean = norm_fit$estimate["mean"],
                        sd   = norm_fit$estimate["sd"]),
      ad_test = ad.test(log_returns,
                        function(x) pnorm(x,
                                          mean = norm_fit$estimate["mean"],
                                          sd   = norm_fit$estimate["sd"])),
      parameters = c(
        xi    = norm_fit$estimate["mean"],
        omega = norm_fit$estimate["sd"],
        alpha = 0,
        nu    = Inf
      )
    )
  })
}

# Robust wrapper: try skew-t, otherwise fall back to Normal.
robust_distribution_fit <- function(log_returns, stock_name) {
  skt_result <- tryCatch(
    fit_skew_t_distribution(log_returns, stock_name),
    error = function(e) NULL
  )
  if (is.null(skt_result)) {
    warning(paste("Falling back to normal distribution for", stock_name))
    return(fit_normal_distribution(log_returns, stock_name))
  }
  return(skt_result)
}

# Fit a Normal distribution by maximum likelihood with KS + AD tests.
fit_normal_distribution <- function(log_returns, stock_name) {
  tryCatch({
    norm_fit <- fitdist(log_returns, "norm")
    mu    <- norm_fit$estimate["mean"]
    sigma <- norm_fit$estimate["sd"]

    ks_test <- ks.test(log_returns, "pnorm", mean = mu, sd = sigma)
    ad_test <- ad.test(log_returns, function(x) pnorm(x, mean = mu, sd = sigma))

    list(
      fit        = norm_fit,
      ks_test    = ks_test,
      ad_test    = ad_test,
      parameters = c(mean = mu, sd = sigma)
    )
  }, error = function(e) {
    warning(paste("Error fitting normal distribution for", stock_name, ":", e$message))
    NULL
  })
}

# -----------------------------------------------------------------------------
# 3. Diagnostic plots (histogram + density, Q-Q, time series, ACF)
# -----------------------------------------------------------------------------
create_distribution_plots <- function(log_returns, stock_name, dist_type, fit_results) {
  par(mfrow = c(2, 2))

  # 1. Histogram with fitted density.
  hist(log_returns, prob = TRUE,
       main = paste(stock_name, "\nLog Returns Distribution (", dist_type, ")"),
       xlab = "Log Returns", breaks = 50)

  if (dist_type == "Skew-t") {
    curve(dst(x,
              xi    = fit_results$parameters["xi"],
              omega = fit_results$parameters["omega"],
              alpha = fit_results$parameters["alpha"],
              nu    = fit_results$parameters["nu"]),
          add = TRUE, col = "red", lwd = 2)
  } else if (dist_type == "Normal") {
    curve(dnorm(x,
                mean = fit_results$parameters["mean"],
                sd   = fit_results$parameters["sd"]),
          add = TRUE, col = "blue", lwd = 2)
  }

  # 2. Q-Q plot.
  if (dist_type == "Skew-t") {
    qqplot(qst(ppoints(length(log_returns)),
               xi    = fit_results$parameters["xi"],
               omega = fit_results$parameters["omega"],
               alpha = fit_results$parameters["alpha"],
               nu    = fit_results$parameters["nu"]),
           log_returns,
           main = paste(stock_name, "\nQ-Q Plot (", dist_type, ")"),
           xlab = "Theoretical Quantiles",
           ylab = "Sample Quantiles")
    abline(0, 1, col = "red")
  } else {
    qqnorm(log_returns, main = paste(stock_name, "\nQ-Q Plot (", dist_type, ")"))
    qqline(log_returns, col = "blue")
  }

  # 3. Log-returns time series.
  plot(log_returns, type = "l",
       main = paste(stock_name, "\nLog Returns Time Series"),
       xlab = "Time", ylab = "Log Returns")

  # 4. Autocorrelation function.
  acf(log_returns, main = paste(stock_name, "\nACF Plot"))
}

# -----------------------------------------------------------------------------
# 4. EDA driver: goodness-of-fit table across all stocks (Normal vs. Skew-t)
# -----------------------------------------------------------------------------
analyze_stock_distributions <- function(stock_data) {
  results_skt  <- list()
  results_norm <- list()

  pdf("stock_distribution_plots.pdf")

  for (stock_name in names(stock_data)[-1]) {  # Exclude 'Date' column
    log_returns <- calculate_log_returns(stock_data[[stock_name]])

    skt_fit <- fit_skew_t_distribution(log_returns, stock_name)
    create_distribution_plots(log_returns, stock_name, "Skew-t", skt_fit)

    norm_fit <- fit_normal_distribution(log_returns, stock_name)
    create_distribution_plots(log_returns, stock_name, "Normal", norm_fit)

    results_skt[[stock_name]] <- list(
      ks_pvalue = skt_fit$ks_test$p.value,
      ad_pvalue = skt_fit$ad_test$p.value,
      aic       = skt_fit$fit$aic,
      bic       = skt_fit$fit$bic
    )
    results_norm[[stock_name]] <- list(
      ks_pvalue = norm_fit$ks_test$p.value,
      ad_pvalue = norm_fit$ad_test$p.value,
      aic       = norm_fit$fit$aic,
      bic       = norm_fit$fit$bic
    )

    cat("\nResults for", stock_name, ":\n")
    cat("\nSkew-t Distribution Tests:\n")
    cat("KS test p-value:", skt_fit$ks_test$p.value, "\n")
    cat("AD test p-value:", skt_fit$ad_test$p.value, "\n")
    cat("AIC:", skt_fit$fit$aic, "\n")
    cat("BIC:", skt_fit$fit$bic, "\n")

    cat("\nNormal Distribution Tests:\n")
    cat("KS test p-value:", norm_fit$ks_test$p.value, "\n")
    cat("AD test p-value:", norm_fit$ad_test$p.value, "\n")
    cat("AIC:", norm_fit$fit$aic, "\n")
    cat("BIC:", norm_fit$fit$bic, "\n")
  }

  dev.off()

  results_df_skt <- data.frame(
    Stock     = names(results_skt),
    KS_pvalue = sapply(results_skt, `[[`, "ks_pvalue"),
    AD_pvalue = sapply(results_skt, `[[`, "ad_pvalue"),
    AIC       = sapply(results_skt, `[[`, "aic"),
    BIC       = sapply(results_skt, `[[`, "bic")
  )
  results_df_norm <- data.frame(
    Stock     = names(results_norm),
    KS_pvalue = sapply(results_norm, `[[`, "ks_pvalue"),
    AD_pvalue = sapply(results_norm, `[[`, "ad_pvalue"),
    AIC       = sapply(results_norm, `[[`, "aic"),
    BIC       = sapply(results_norm, `[[`, "bic")
  )

  # Flag fits rejected at the 5% level.
  results_df_skt$KS_Significance  <- ifelse(results_df_skt$KS_pvalue  < 0.05, "*", "")
  results_df_skt$AD_Significance  <- ifelse(results_df_skt$AD_pvalue  < 0.05, "*", "")
  results_df_norm$KS_Significance <- ifelse(results_df_norm$KS_pvalue < 0.05, "*", "")
  results_df_norm$AD_Significance <- ifelse(results_df_norm$AD_pvalue < 0.05, "*", "")

  list(
    skew_t_results = results_df_skt,
    normal_results = results_df_norm
  )
}

# -----------------------------------------------------------------------------
# 5. Bayesian hierarchical Skew-t model (Gibbs sampling via JAGS)
# -----------------------------------------------------------------------------
#
# Hierarchical specification (per stock j, observation i):
#   y[i,j] ~ Normal(mu[j] + alpha[j] * z[i,j], tau[j])
#   z[i,j] ~ HalfNormal             # latent variable injecting skewness
#   mu[j]    ~ Normal(mu0, tau0)
#   tau[j]   ~ Gamma(a0, b0);  sigma[j] = 1/sqrt(tau[j])
#   alpha[j] ~ Normal(0, 1)
#   nu       ~ Gamma(2, 0.1)
#   mu0 ~ Normal(0,1); tau0 ~ Gamma(1e-4, 0.05); a0,b0 ~ Gamma(0.1, 0.5)
#
perform_bayesian_analysis_skewt <- function(stock_data) {
  log_returns_list <- lapply(selected_stocks, function(stock_name) {
    calculate_log_returns(stock_data[[stock_name]])
  })

  # Align series to common length.
  min_length  <- min(sapply(log_returns_list, length))
  log_returns <- do.call(cbind, lapply(log_returns_list, function(x) x[1:min_length]))

  # JAGS model: skew-t approximated by a half-normal latent mixture.
  writeLines("model {
    # Likelihood for each stock's log returns
    for (i in 1:N) {
      for (j in 1:nstock) {
        y[i,j] ~ dnorm(mu[j] + alpha[j] * z[i,j], tau[j])
        z[i,j] ~ dnorm(0, 2) T(0,)  # Half-normal latent variable (skewness)
      }
    }

    # Hierarchical priors
    for (j in 1:nstock) {
      mu[j]    ~ dnorm(mu0, tau0)       # Location
      tau[j]   ~ dgamma(a0, b0)         # Precision
      sigma[j] <- 1 / sqrt(tau[j])      # Scale (derived)
      alpha[j] ~ dnorm(0, 1)            # Skewness
    }

    # Degrees of freedom (prior only; see header note)
    nu ~ dgamma(2, 0.1)

    # Hyperpriors
    mu0  ~ dnorm(0, 1)
    tau0 ~ dgamma(0.0001, 0.05)
    a0   ~ dgamma(0.1, 0.5)
    b0   ~ dgamma(0.1, 0.5)
  }", con = "skewt_model.bug")

  jags_data <- list(
    N      = nrow(log_returns),
    nstock = ncol(log_returns),
    y      = log_returns
  )

  # Three overdispersed chains for the Gelman-Rubin diagnostic.
  inits <- list(
    list(mu = apply(log_returns, 2, mean),
         tau = 1 / apply(log_returns, 2, var),
         alpha = rep(0, ncol(log_returns)),
         nu = 5, mu0 = 0, tau0 = 1, a0 = 1, b0 = 1),
    list(mu = apply(log_returns, 2, mean) + 0.01,
         tau = 1 / apply(log_returns, 2, var) + 0.001,
         alpha = rep(0.1, ncol(log_returns)),
         nu = 7, mu0 = 1, tau0 = 1.1, a0 = 1.2, b0 = 1.1),
    list(mu = apply(log_returns, 2, mean) - 0.01,
         tau = pmax(1 / apply(log_returns, 2, var) - 0.001, 0.001),
         alpha = rep(-0.1, ncol(log_returns)),
         nu = 6, mu0 = -1, tau0 = 0.9, a0 = 0.8, b0 = 0.9)
  )

  set.seed(123)
  jags_model <- jags.model(
    file     = "skewt_model.bug",
    data     = jags_data,
    inits    = inits,
    n.chains = 3,
    n.adapt  = 30000
  )

  update(jags_model, n.iter = 50000)  # Burn-in

  samples <- coda.samples(
    model          = jags_model,
    variable.names = c("mu", "sigma", "alpha", "nu", "mu0", "tau0"),
    n.iter         = 150000,
    thin           = 2
  )

  summary_samples <- summary(samples)
  dic <- dic.samples(jags_model, n.iter = 10000)

  bayesian_results <- data.frame(
    Stock = selected_stocks,
    Mu    = summary_samples$statistics[grep("^mu\\[",    rownames(summary_samples$statistics)), "Mean"],
    Sigma = summary_samples$statistics[grep("^sigma\\[", rownames(summary_samples$statistics)), "Mean"],
    Alpha = summary_samples$statistics[grep("^alpha\\[", rownames(summary_samples$statistics)), "Mean"],
    Nu    = summary_samples$statistics["nu", "Mean"]
  )

  # Stack posterior draws across chains.
  mu_samples    <- do.call(rbind, lapply(samples, function(x) x[, grep("^mu\\[",    colnames(x))]))
  sigma_samples <- do.call(rbind, lapply(samples, function(x) x[, grep("^sigma\\[", colnames(x))]))
  alpha_samples <- do.call(rbind, lapply(samples, function(x) x[, grep("^alpha\\[", colnames(x))]))
  nu_samples    <- do.call(rbind, lapply(samples, function(x) x[, "nu"]))

  weights <- rep(1 / ncol(log_returns), ncol(log_returns))  # Equal-weight portfolio

  portfolio_mu    <- rowSums(mu_samples    * matrix(weights, nrow = nrow(mu_samples),    ncol = ncol(mu_samples),    byrow = TRUE))
  portfolio_sigma <- sqrt(rowSums((sigma_samples * matrix(weights, nrow = nrow(sigma_samples), ncol = ncol(sigma_samples), byrow = TRUE))^2))
  portfolio_alpha <- rowSums(alpha_samples * matrix(weights, nrow = nrow(alpha_samples), ncol = ncol(alpha_samples), byrow = TRUE))

  portfolio_stats <- list(
    mean     = mean(portfolio_mu),
    sd       = mean(portfolio_sigma),
    skewness = mean(portfolio_alpha),
    nu       = mean(nu_samples),
    var_5    = quantile(portfolio_mu - 1.645 * portfolio_sigma, 0.05),  # Approximate VaR
    var_95   = quantile(portfolio_mu + 1.645 * portfolio_sigma, 0.95)
  )

  list(
    bayesian_results = bayesian_results,
    portfolio_stats  = portfolio_stats,
    dic              = dic,
    mcmc_samples     = samples,
    convergence      = gelman.diag(samples)
  )
}

# -----------------------------------------------------------------------------
# 6. MCMC diagnostics (trace/density, ACF, Gelman-Rubin, Geweke, Heidel, ESS)
# -----------------------------------------------------------------------------
plot_mcmc_diagnostics_skewt <- function(bayesian_analysis) {
  pdf("mcmc_diagnostics_skewt.pdf")

  plot(bayesian_analysis$mcmc_samples)  # Trace + density

  for (i in seq_along(bayesian_analysis$mcmc_samples)) {
    acf(as.matrix(bayesian_analysis$mcmc_samples[[i]]),
        main = paste("Chain", i, "Autocorrelation"))
  }
  pairs(as.matrix(bayesian_analysis$mcmc_samples[[1]]))  # Cross-correlation

  dev.off()

  cat("\nGelman-Rubin Convergence Diagnostic:\n")
  print(bayesian_analysis$convergence)

  cat("\nHeidelberger and Welch Convergence Diagnostic:\n")
  print(heidel.diag(bayesian_analysis$mcmc_samples))

  cat("\nGeweke Convergence Diagnostic:\n")
  print(geweke.diag(bayesian_analysis$mcmc_samples))

  cat("\nEffective Sample Size:\n")
  print(effectiveSize(bayesian_analysis$mcmc_samples))
}

# -----------------------------------------------------------------------------
# 7. Copula analysis (in-sample): fit 5 copulas, simulate portfolio returns
# -----------------------------------------------------------------------------
perform_copula_analysis_skewt <- function(stock_data) {
  log_returns_list <- lapply(selected_stocks, function(stock_name) {
    calculate_log_returns(stock_data[[stock_name]])
  })
  min_length  <- min(sapply(log_returns_list, length))
  log_returns <- do.call(cbind, lapply(log_returns_list, function(x) x[1:min_length]))

  # Empirical-CDF (pseudo-observations) for standard copulas.
  uniform_transforms <- apply(log_returns, 2, function(x) rank(x) / (length(x) + 1))

  copula_types <- list(
    gaussian = normalCopula(dim = ncol(log_returns), dispstr = "un"),
    t        = tCopula(dim = ncol(log_returns), dispstr = "un", df = 4, df.fixed = TRUE),
    skewt    = NULL,  # Fitted separately below
    clayton  = claytonCopula(dim = ncol(log_returns)),
    gumbel   = gumbelCopula(dim = ncol(log_returns))
  )

  copula_fits <- list()

  # Fit Gaussian, t, Clayton, Gumbel by ML.
  for (cop_name in names(copula_types)[c(1, 2, 4, 5)]) {
    tryCatch({
      fit <- fitCopula(copula_types[[cop_name]], data = uniform_transforms, method = "ml")
      aic <- 2 * length(coef(fit)) - 2 * logLik(fit)
      fitted_copula <- fit@copula
      fitted_copula@parameters <- coef(fit)
      copula_fits[[cop_name]] <- list(name = cop_name, AIC = aic, copula = fitted_copula)
    }, error = function(e) {
      warning(paste("Failed to fit", cop_name, "copula:", e$message))
    })
  }

  # Skew-t margins -> skew-t copula space.
  skewt_fits   <- lapply(seq_len(ncol(log_returns)),
                         function(i) fit_skew_t_distribution(log_returns[, i], selected_stocks[i]))
  skewt_params <- lapply(skewt_fits, function(fit) fit$parameters)

  skewt_uniforms <- matrix(NA, nrow = nrow(log_returns), ncol = ncol(log_returns))
  for (i in seq_len(ncol(log_returns))) {
    p <- skewt_params[[i]]
    skewt_uniforms[, i] <- pst(log_returns[, i],
                               xi = p["xi"], omega = p["omega"],
                               alpha = p["alpha"], nu = p["nu"])
  }

  tryCatch({
    skewt_cop <- tCopula(dim = ncol(log_returns), dispstr = "un")
    skewt_fit <- fitCopula(skewt_cop, data = skewt_uniforms, method = "ml")
    aic <- 2 * (length(coef(skewt_fit)) + 4 * ncol(log_returns)) -
      2 * (logLik(skewt_fit) + sum(sapply(skewt_fits, function(x) x$fit$loglik)))
    fitted_copula <- skewt_fit@copula
    fitted_copula@parameters <- coef(skewt_fit)
    copula_fits[["skewt"]] <- list(name = "skewt", AIC = aic, copula = fitted_copula,
                                   marginal_params = skewt_params)
  }, error = function(e) {
    warning("Failed to fit skewed-t copula:", e$message)
  })

  # Best copula by AIC.
  aic_values  <- sapply(copula_fits, function(x) x$AIC)
  best_copula <- copula_fits[[which.min(aic_values)]]

  # Monte Carlo simulation of the joint portfolio distribution.
  n_samples <- 10000
  if (best_copula$name == "skewt") {
    simulated_uniform <- rCopula(n_samples, best_copula$copula)
    simulated_returns <- matrix(NA, nrow = n_samples, ncol = ncol(log_returns))
    for (i in seq_len(ncol(log_returns))) {
      p <- best_copula$marginal_params[[i]]
      simulated_returns[, i] <- qst(simulated_uniform[, i],
                                    xi = p["xi"], omega = p["omega"],
                                    alpha = p["alpha"], nu = p["nu"])
    }
  } else {
    simulated_uniform <- rCopula(n_samples, best_copula$copula)
    simulated_returns <- apply(simulated_uniform, 2, function(u) {
      quantile(log_returns, probs = u, na.rm = TRUE)
    })
  }

  portfolio_returns <- rowMeans(simulated_returns)  # Equal weights
  portfolio_stats <- list(
    mean     = mean(portfolio_returns),
    variance = var(portfolio_returns),
    var_5    = quantile(portfolio_returns, 0.05),
    var_95   = quantile(portfolio_returns, 0.95),
    skewness = skewness(portfolio_returns),
    kurtosis = kurtosis(portfolio_returns)
  )

  list(
    copula_comparison = data.frame(
      Copula = names(copula_fits),
      AIC    = sapply(copula_fits, function(x) x$AIC)
    ),
    best_copula         = best_copula$name,
    portfolio_returns   = portfolio_stats,
    marginal_parameters = if (best_copula$name == "skewt") best_copula$marginal_params else NULL,
    original_returns    = log_returns
  )
}

# -----------------------------------------------------------------------------
# 8. Copula out-of-sample validation (train on a held-in window, test on rest)
# -----------------------------------------------------------------------------
# NOTE: train_proportion = 0.2 fits each copula on the first 20% of the sample
# and scores test log-likelihood on the remaining 80%. This is intentional but
# unconventional (most validation trains on the majority); see README.
perform_copula_out_of_sample_validation <- function(stock_data, train_proportion = 0.2) {
  log_returns_list <- lapply(selected_stocks, function(stock_name) {
    calculate_log_returns(stock_data[[stock_name]])
  })
  min_length  <- min(sapply(log_returns_list, length))
  log_returns <- do.call(cbind, lapply(log_returns_list, function(x) x[1:min_length]))

  train_size        <- floor(min_length * train_proportion)
  log_returns_train <- log_returns[1:train_size, ]
  log_returns_test  <- log_returns[(train_size + 1):min_length, ]

  copula_types <- list(
    gaussian = normalCopula(dim = ncol(log_returns), dispstr = "un"),
    t        = tCopula(dim = ncol(log_returns), dispstr = "un", df = 4, df.fixed = TRUE),
    skewt    = NULL,
    clayton  = claytonCopula(dim = ncol(log_returns)),
    gumbel   = gumbelCopula(dim = ncol(log_returns))
  )

  uniform_transforms_train <- apply(log_returns_train, 2, function(x) rank(x) / (length(x) + 1))
  uniform_transforms_test  <- apply(log_returns_test,  2, function(x) rank(x) / (length(x) + 1))

  copula_fits      <- list()
  model_validation <- list()

  for (cop_name in names(copula_types)[c(1, 2, 4, 5)]) {
    tryCatch({
      fit <- fitCopula(copula_types[[cop_name]], data = uniform_transforms_train, method = "ml")
      aic <- 2 * length(coef(fit)) - 2 * logLik(fit)
      fitted_copula <- fit@copula
      fitted_copula@parameters <- coef(fit)
      copula_fits[[cop_name]] <- list(name = cop_name, AIC = aic, copula = fitted_copula)

      test_ll <- sum(log(dCopula(uniform_transforms_test, fitted_copula)))
      model_validation[[cop_name]] <- test_ll
    }, error = function(e) {
      warning(paste("Failed to fit", cop_name, "copula:", e$message))
    })
  }

  # Skew-t margins fitted on the training window.
  skewt_fits   <- lapply(seq_len(ncol(log_returns_train)),
                         function(i) fit_skew_t_distribution(log_returns_train[, i], selected_stocks[i]))
  skewt_params <- lapply(skewt_fits, function(fit) fit$parameters)

  skewt_uniforms_train <- matrix(NA, nrow = nrow(log_returns_train), ncol = ncol(log_returns_train))
  for (i in seq_len(ncol(log_returns_train))) {
    p <- skewt_params[[i]]
    skewt_uniforms_train[, i] <- pst(log_returns_train[, i],
                                     xi = p["xi"], omega = p["omega"],
                                     alpha = p["alpha"], nu = p["nu"])
  }
  skewt_uniforms_test <- matrix(NA, nrow = nrow(log_returns_test), ncol = ncol(log_returns_test))
  for (i in seq_len(ncol(log_returns_test))) {
    p <- skewt_params[[i]]
    skewt_uniforms_test[, i] <- pst(log_returns_test[, i],
                                    xi = p["xi"], omega = p["omega"],
                                    alpha = p["alpha"], nu = p["nu"])
  }

  tryCatch({
    skewt_cop <- tCopula(dim = ncol(log_returns_train), dispstr = "un")
    skewt_fit <- fitCopula(skewt_cop, data = skewt_uniforms_train, method = "ml")
    aic <- 2 * (length(coef(skewt_fit)) + 4 * ncol(log_returns_train)) -
      2 * (logLik(skewt_fit) + sum(sapply(skewt_fits, function(x) x$fit$loglik)))
    fitted_copula <- skewt_fit@copula
    fitted_copula@parameters <- coef(skewt_fit)
    copula_fits[["skewt"]] <- list(name = "skewt", AIC = aic, copula = fitted_copula,
                                   marginal_params = skewt_params)

    test_ll <- sum(log(dCopula(skewt_uniforms_test, fitted_copula)))
    model_validation[["skewt"]] <- test_ll
  }, error = function(e) {
    warning("Failed to fit skewed-t copula:", e$message)
  })

  ll_values        <- unlist(model_validation)
  best_copula_name <- names(ll_values)[which.max(ll_values)]
  best_copula      <- copula_fits[[best_copula_name]]

  list(
    copula_comparison = data.frame(
      Copula             = names(copula_fits),
      AIC                = sapply(copula_fits, function(x) x$AIC),
      Test_LogLikelihood = ll_values
    ),
    best_copula         = best_copula_name,
    train_proportion    = train_proportion,
    marginal_parameters = if (best_copula_name == "skewt") best_copula$marginal_params else NULL
  )
}

# -----------------------------------------------------------------------------
# 9. Main driver
# -----------------------------------------------------------------------------
main <- function() {
  stock_data <- read_csv(data_path)

  # Per-stock diagnostic plots (skew-t and normal).
  for (stock_name in names(stock_data)[-1]) {  # Exclude 'Date' column
    log_returns <- calculate_log_returns(stock_data[[stock_name]])
    skt_fit  <- fit_skew_t_distribution(log_returns, stock_name)
    create_distribution_plots(log_returns, stock_name, "Skew-t", skt_fit)
    norm_fit <- fit_normal_distribution(log_returns, stock_name)
    create_distribution_plots(log_returns, stock_name, "Normal", norm_fit)
  }

  # Part 1: goodness-of-fit summary.
  results <- analyze_stock_distributions(stock_data)
  cat("\nFinal Summary:\n")
  cat("\nSkew-t Distribution Results:\n"); print(results$skew_t_results)
  cat("\nNormal Distribution Results:\n"); print(results$normal_results)

  # Part 2: Bayesian hierarchical skew-t model + diagnostics.
  bayesian_analysis <- perform_bayesian_analysis_skewt(stock_data)
  plot_mcmc_diagnostics_skewt(bayesian_analysis)
  cat("\nBayesian Analysis Results:\n"); print(bayesian_analysis$bayesian_results)
  cat("\nPortfolio Statistics:\n");      print(bayesian_analysis$portfolio_stats)
  cat("\nDIC:\n");                        print(bayesian_analysis$dic)

  # Part 3: copula analysis (in-sample).
  copula_analysis <- perform_copula_analysis_skewt(stock_data)
  cat("\nCopula Model Comparison:\n");          print(copula_analysis$copula_comparison)
  cat("\nBest Copula Model:", copula_analysis$best_copula, "\n")
  cat("\nPortfolio Returns Statistics:\n");      print(copula_analysis$portfolio_returns)
  if (!is.null(copula_analysis$marginal_parameters)) {
    cat("\nSkewed-t Parameters for Each Stock:\n"); print(copula_analysis$marginal_parameters)
  }

  # Part 4: copula out-of-sample validation.
  out_of_sample_results <- perform_copula_out_of_sample_validation(stock_data)
  cat("\nOut-of-Sample Copula Model Validation:\n"); print(out_of_sample_results$copula_comparison)
  cat("\nBest Copula Model (Out-of-Sample):", out_of_sample_results$best_copula, "\n")
  cat("Training Proportion:", out_of_sample_results$train_proportion, "\n")
}

# Run end-to-end.
main()

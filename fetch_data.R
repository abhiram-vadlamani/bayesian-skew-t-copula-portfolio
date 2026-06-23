# =============================================================================
# fetch_data.R -- regenerate data/stock_data.csv from Yahoo Finance
# -----------------------------------------------------------------------------
# The analysis in bayesian_skew_t_copula.R expects a CSV with a Date column
# followed by one adjusted-close price column per ticker. The original study
# used ~500 trading days (roughly two years ending in late 2024). Yahoo
# back-adjusts historical prices, so a fresh pull may differ slightly from the
# exact figures in the report.
# =============================================================================

required <- c("quantmod", "readr")
new_pkgs <- required[!(required %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)
invisible(lapply(required, library, character.only = TRUE))

tickers    <- c("MSFT", "JNJ", "PG", "PEP", "WMT", "TSLA", "NVDA", "AMD", "AMZN", "META")
start_date <- as.Date("2022-12-01")
end_date   <- as.Date("2024-12-01")

prices <- lapply(tickers, function(tk) {
  message("Downloading ", tk, " ...")
  px <- getSymbols(tk, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
  Ad(px)  # Adjusted close
})

merged <- Reduce(function(a, b) merge(a, b, join = "inner"), prices)
colnames(merged) <- tickers

out <- data.frame(Date = index(merged), coredata(merged))

dir.create("data", showWarnings = FALSE)
write_csv(out, "data/stock_data.csv")
message("Wrote data/stock_data.csv  (", nrow(out), " rows x ", length(tickers), " tickers)")

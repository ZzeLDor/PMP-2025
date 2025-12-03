import numpy as np
import pymc as pm
import arviz as az

publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0, 15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])
new_publicity = np.array([2.8, 7.8, 12.0])

with pm.Model() as model:
    x = pm.Data("publicity", publicity)
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    mu = alpha + beta * x
    sales_obs = pm.Normal("sales_obs", mu=mu, sigma=sigma, observed=sales)
    trace = pm.sample(2000, tune=1000, target_accept=0.9)

credhdi = az.hdi(trace, var_names=["alpha", "beta"], hdi_prob=0.95)
ahdi = credhdi["alpha"].values
bhdi = credhdi["beta"].values
print(f"\nIntercept HDI: {ahdi}")
print(f"Slope HDI: {bhdi}\n")

with model:
    pm.set_data({"publicity": new_publicity})
    sales_pred = pm.Normal("sales_pred", mu=mu, sigma=sigma)
    pp = pm.sample_posterior_predictive(trace, var_names=["sales_pred"])

pp_vals = pp.posterior_predictive["sales_pred"].values
pred_int = np.percentile(pp_vals, [5, 95], axis=(0, 1))

for h, lo, hi in zip(new_publicity, pred_int[0], pred_int[1]):
    print(f"Publicity = {h} => sales ~ {lo:.2f} to {hi:.2f}")
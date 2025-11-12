import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
np.random.seed(12)

dobs = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

# a) Pentru x in N(x, 10^2), alegem media datelor observate, e logic sa centram a priori-ul in jurul valorilor pe care le vedem
x = np.mean(dobs)
with pm.Model() as model_ps:
    mu = pm.Normal("mu", mu=x, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=dobs)
    ps_trace = pm.sample(2000, return_inferencedata=True, random_seed=12)

# b)
summary_ps = az.summary(ps_trace, hdi_prob=0.95)
print(summary_ps)
print()

hdi_mu_ps = az.hdi(ps_trace, hdi_prob=0.95)["mu"].values
hdi_sigma_ps = az.hdi(ps_trace, hdi_prob=0.95)["sigma"].values

print("HDI 95% pentru mu:", hdi_mu_ps)
print("HDI 95% pentru sigma:", hdi_sigma_ps)
print()
print()

# c)
freq_mean = np.mean(dobs)
freq_std = np.std(dobs)

posterior_mu_ps = ps_trace.posterior["mu"].values.flatten()
posterior_sigma_ps = ps_trace.posterior["sigma"].values.flatten()

print("Media frecventista:", freq_mean)
print("Deviatia standard frecventista:", freq_std)
print()
print("Media posterioara mu:", np.mean(posterior_mu_ps))
print("Media posterioara sigma:", np.mean(posterior_sigma_ps))
print()
print()
# Se observa ca diferentele sunt foarte mici datorita a priori-ului slab. Daaca am avea si mai multe date, diferenteele ar fi si mai mici.

# d)
with pm.Model() as model_pp:
    mu = pm.Normal("mu", mu=50, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=10)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=dobs)
    pp_trace = pm.sample(2000, return_inferencedata=True, random_seed=12)

pp_sum = az.summary(pp_trace, hdi_prob=0.95)
print(pp_sum)
print()

posterior_mu_pp = pp_trace.posterior["mu"].values.flatten()
posterior_sigma_pp = pp_trace.posterior["sigma"].values.flatten()

hdi_mu_pp = az.hdi(pp_trace, hdi_prob=0.95)["mu"].values
hdi_sigma_pp = az.hdi(pp_trace, hdi_prob=0.95)["sigma"].values

print("HDI 95% pentru mu:", hdi_mu_pp)
print("HDI 95% pentru sigma:", hdi_sigma_pp)
print()
print()
print("Model a) - mu posterior:", np.mean(posterior_mu_ps))
print("Model d) - mu posterior:", np.mean(posterior_mu_pp))
print()
print("Model a) - sigma posterior:", np.mean(posterior_sigma_ps))
print("Model d) - sigma posterior:", np.mean(posterior_sigma_pp))

# A priori-ul puternic trage estimarea lui mu spre 50, desi datele observate au media in jur de 58. Cand vine vorba de sigma, a priori-ul e la fel in ambele modele, deci estimarea nu se schimba la fel de mult.

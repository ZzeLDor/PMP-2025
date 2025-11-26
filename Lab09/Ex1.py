import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

yobs = [0, 5, 10]
thsi = [0.2, 0.5]

fig_post, axes_post = plt.subplots(2, 3, figsize=(18, 10))
fig_post.suptitle('Distributia Posterioara pentru n', fontsize=16)

fig_pred, axes_pred = plt.subplots(2, 3, figsize=(18, 10))
fig_pred.suptitle('Distributia Predictiva Posterioara pentru Y*', fontsize=16)

for i, th in enumerate(thsi):
    for j, y_obs in enumerate(yobs):
        with pm.Model() as model_magazin:
            n = pm.Poisson("n", mu=10)
            obs = pm.Binomial("obs", n=n, p=th, observed=y_obs)
            trace = pm.sample(1000, return_inferencedata=True)
            pp = pm.sample_posterior_predictive(trace)

        ax_curr = axes_post[i, j]
        az.plot_posterior(trace, var_names=["n"], ax=ax_curr, hdi_prob=0.95)
        ax_curr.set_title(f"Theta={th}, Y_obs={y_obs}")
        ax_curr.set_xlabel("n (nr. clienti)")

        ax_pred = axes_pred[i, j]
        y_pred_samples = pp.posterior_predictive["obs"].values.flatten()

        az.plot_dist(y_pred_samples, ax=ax_pred)
        ax_pred.set_title(f"Theta={th}, Y_obs={y_obs}")
        ax_pred.set_xlabel("Y* (cumparatori prezisi)")
plt.show()

# La b), observam ca pe cat creste Y, pe atat creste si n; daca avem mai multi cumparatori, automat avem si mai multi clienti din care sa se traga cumparatorii
# Dar theta este invers proportional cu numarul de clienti n; cu cat creste probabilitatea ca un client sa cumpere, cu atat mai putin nevooie avem de clienti ca sa ajungem la Y cumparatori

# La d), distributia predictiva posterioara pentru Y* arata cati cumparatori vor fi in viitor pe baza datelor pe care le avem, iar posteriorul pentru n estimeaza cati clienti avem bazat de datele Y si theta
# Pentru DPP pentru Y*, pe cat theta si Y cresc, pe atat Y* creste
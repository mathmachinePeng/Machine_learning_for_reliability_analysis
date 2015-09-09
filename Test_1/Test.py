
import pymc as pm
import matplotlib
from matplotlib import pyplot as plt

p = pm.Uniform('p', lower = 0, upper = 1)
p_true = 0.05
N = 1500

occurrences = pm.rbernoulli(p_true, N)

print occurrences
print occurrences.sum()

print "What is the observed frequency in Group A? %.4f" % occurrences.mean()
print "Does this equal the true frequency? %s" % (occurrences.mean() == p_true)

obs = pm.Bernoulli("obs", p, value = occurrences, observed = True)

mcmc = pm.MCMC([p, obs])
mcmc.sample(18000, 1000)


plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
plt.hist(mcmc.trace("p")[:], bins=25, histtype="stepfilled", normed=True)
plt.legend()

plt.show()
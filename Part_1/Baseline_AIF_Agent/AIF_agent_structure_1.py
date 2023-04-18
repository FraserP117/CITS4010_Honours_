import numpy as np
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt


# The generative model is the model that generates the agent's sensory input.
# In Active Inference, the generative model is represented by a probabilistic
# graphical model. We can define the generative model using Pyro's modeling
# language.

# In this example, the generative model has two hidden states (represented by
# the variable z) and two observed states (represented by the variable x).
# The generative model defines a prior distribution over the hidden states
# (a normal distribution with mean 0 and variance 1) and a mapping from the
# hidden states to the observed states (a normal distribution with mean z^3
# and variance 0.1).
def generative_model(z, x):
    # Prior over hidden states
    z_prior = torch.tensor([0.0, 0.0])
    z = pyro.sample("z", dist.Normal(z_prior, 1.0))

    # Mapping from hidden states to observations
    x_mean = torch.tensor([z[0] ** 3, z[1] ** 3])
    x = pyro.sample("x", dist.Normal(x_mean, 0.1))

    return x

# Define the agent's beliefs
# The agent's beliefs are represented by a second probabilistic graphical model
# that is conditioned on the agent's sensory input. We can define the agent's
# beliefs using Pyro's modeling language.

# In this example, the agent's beliefs also have two hidden states
# (represented by the variable z) and two observed states
# (represented by the variable x). The agent's beliefs define a prior
# distribution over the hidden states (a normal distribution with mean 0 and
# variance 1) and a mapping from the hidden states to the observed states
# (a normal distribution with mean z^3 and variance 0.1). The agent's beliefs
# also include a precision weighting of the sensory prediction error
# (represented by the variable kappa) and a prediction error
def belief_model(x):
    # Prior over hidden states
    z_prior = torch.tensor([0.0, 0.0])
    z = pyro.sample("z", dist.Normal(z_prior, 1.0))
    # Mapping from hidden states to observations
    x_mean = torch.tensor([z[0] ** 3, z[1] ** 3])
    # Precision weighting of sensory prediction error
    kappa = 1.0
    # Prediction error
    pe = (x - x_mean) * kappa
    # Free energy
    fe = 0.5 * torch.sum(pe ** 2)
    pyro.sample("x", dist.Normal(x_mean, 0.1), obs=x)
    return fe


# To simulate interactions with an environment using the Active Inference model
# described above, we can use Pyro's inference engine to infer the hidden states
# of the generative model based on the observed states, and then update the
# agent's beliefs based on the inferred hidden states. We can then repeat this
# process for multiple time steps to simulate interactions with the environment.

# Set up the environment
num_steps = 100
true_z = np.zeros((num_steps, 2))
obs_x = np.zeros((num_steps, 2))
for t in range(num_steps):
    true_z[t] = np.random.normal(0, 1, size=2)
    obs_x[t] = np.random.normal(true_z[t]**3, 0.1)

# Set up the initial beliefs
beliefs = np.zeros((num_steps, 2))
beliefs[0] = np.random.normal(0, 1, size=2)

# Simulate interactions with the environment
for t in range(1, num_steps):
    # Infer the hidden states of the generative model
    conditioned_model = pyro.condition(generative_model, data={"x": torch.tensor(obs_x[t])})
    inferred_model = pyro.infer.Importance(conditioned_model, num_samples=100)
    # posterior = inferred_model.run(torch.tensor(beliefs[t-1]))
    posterior = inferred_model.run(torch.tensor(beliefs[t-1]), torch.tensor(obs_x[t-1]))

    # Update the agent's beliefs based on the inferred hidden states
    conditioned_belief = pyro.condition(belief_model, data={"z": posterior})
    inferred_belief = pyro.infer.Importance(conditioned_belief, num_samples=100)
    beliefs[t] = inferred_belief.run(torch.tensor(obs_x[t])).mean.detach().numpy()

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(true_z[:, 0], label="True z1")
ax[0].plot(beliefs[:, 0], label="Inferred z1")
ax[0].legend()
ax[0].set_ylabel("z1")
ax[1].plot(true_z[:, 1], label="True z2")
ax[1].plot(beliefs[:, 1], label="Inferred z2")
ax[1].legend()
ax[1].set_ylabel("z2")
ax[1].set_xlabel("Time")
plt.show()

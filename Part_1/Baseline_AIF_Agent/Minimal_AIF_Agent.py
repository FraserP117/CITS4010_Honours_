import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scipy.stats import randint
from scipy.special import softmax

class State(object):

    def __init__(self, location, chemical_gradient):
        self.location = location
        self.chemical_gradient = chemical_gradient

class ring_environment(object):

    def __init__(self, size):
        self.arena = [State(i, 0.0) for i in range(size)]
        self.size = size

    def __str__(self):
        return f"{[(state.location, state.chemical_gradient) for state in self.arena]}"

    def place_chemical(self, location):
        '''
        Place chemical source at location 'location', with value of 'self.size'
        '''

        chemical_quantity = self.size // 2

        if location > self.size or location < 0:
            return
        else:

            current_location = location

            for i in range(self.size):
                self.arena[(current_location + i) % self.size].chemical_gradient = abs(chemical_quantity)
                self.arena[(current_location - i) % self.size].chemical_gradient = abs(chemical_quantity)
                chemical_quantity -= 1

    def step(self, action):
        # return an observation
        pass

    def reset(self):
        pass

class Agent(object):

    def __init__(self, initial_position, state_space):
        self.position = initial_position
        self.brain_states = randint.stats(0, state_space) # prior on P(phi | b)

    # def update_brain_state(self, next_brain_state):
    #     return softmax(next_brain_state)

    def sensory_dynamics(psi):
        """
        P(s | psi)
        Probability of experiencing sensory state 1 given current position.
        """
        return MAX_SENSE_PROBABILITY * np.exp(
            -SENSE_DECAY_RATE * \
            np.minimum(
                np.abs(psi - FOOD_POSITION),
                np.abs(psi - FOOD_POSITION - ENV_SIZE)
            )
        )


    def model_encoding(self, b):
        """
        Probability of occupying specific position as encoded in the internal state.
        """
        # Softmax function. The shift by b.max() is for numerical stability
        return np.exp(b - b.max()) / np.sum(np.exp(b - b.max()))


    def model_encoding_derivative(self, b):
        """
        Derivative of the model encoding for free energy gradient calculation
        """
        softmax_b = model_encoding(b)
        # Softmax derivative
        return np.diag(softmax_b) - np.outer(softmax_b, softmax_b)


    def generative_density(self, b, a):
        """
        P(psi', s | b, a)
        Agent's prediction of its new position given its internal state and selected action
        (calculated separately for two sensory states).

        P(psi', s | b, a) = Sum_over_psi(P(psi' | s, b, a, psi) * P(s | b, a, psi) * P(psi | b, a))
                          = Sum_over_psi(P(psi' | a, psi) * P(s | psi) * P(psi | b))

        since psi' only depends on a and psi, s only depends on psi, and psi only depends on b.

        """
        # sensory dynamics for each position:
        """P(s | psi)"""
        sd = sensory_dynamics(np.arange(ENV_SIZE))

        """
        P(psi', s | b, a) = Sum_over_psi(P(psi' | a, psi) * P(s | psi) * P(psi | b))
        Note that the Sum_over_psi is only taken over the two positions psi that can result in getting to psi' given a
        """
        psi_prime_0 = ((1-MOVEMENT_PROBABILITY) * (1 - sd) * model_encoding(b) +
                     MOVEMENT_PROBABILITY * np.roll((1 - sd) * model_encoding(b), a))

        psi_prime_1 = ((1-MOVEMENT_PROBABILITY) * sd * model_encoding(b) +
                     MOVEMENT_PROBABILITY * np.roll(sd * model_encoding(b), a))

        return [psi_prime_0, psi_prime_1]


    def variational_density(self, b):
        """
        P(psi | b)
        Agent's belief about the external states (i.e. its current position in the
        world) or intention (i.e. desired position in the world) as encoded in the
        internal state.
        """
        return model_encoding(b)


    def KL(self, a, b):
        """
        Kullback-Leibler divergence between densities a and b.
        """
        return np.sum(a * (np.log(a) - np.log(b)))


    def free_energy(self, b_star, b, s, a):
        """
        KL divergence between variational density and generative density for a fixed
        sensory state s.
        """
        return KL(variational_density(b_star), generative_density(b, a)[s])


    def fe_gradient(self, b_star, b, s, a):
        """
        Partial derivatives of the free energy with respect to belief R1.
        """
        J = model_encoding_derivative(b_star)
        Y = np.log(variational_density(b_star) / generative_density(b, a)[s])
        return np.dot(J,Y)


location = 3

ring = ring_environment(16)
print(ring)
ring.place_chemical(location)
print(f"placed at location: {location}")
print(ring)

agent = Agent(initial_position = 0, state_space = ring.size)

import numpy as np
import pytorch as torch
import gym


class Agent(object):

    def __init__(self, lr, n_actions, state_space, epsilon, eps_start, eps_end, eps_dec, gamma, iters):
        self.lr = lr # learning rate
        self.actions = [i for i in range(self.n_actions)],
        self.state_space = state_space,
        self.epsilon = epsilon
        self.eps_start = eps_start,
        self.eps_end = eps_end,
        self.eps_dec = eps_dec,
        self.gamma = gamma
        self.iters = iters

        self.Q = {}
        self.init_Q()

    def init_Q(self):
        for state in self.state_space:
            for action in self.actions:
                self.Q[(state, action)] = 0.0

    def optimal_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in self.action_space]) # all action values for the current state
        optimal_action = np.argmax(actions)

        return optimal_action

    def select_action(self, state):
        '''
        Epsilon-greedy action-selection
        '''
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.optimal_action(state)

        return action

    def logistic(self, x, L, k, x_m):
    	'''
    	x:   the domain value
    	L:   the supremmum
    	k:   growth rate
    	x_m: x-value of the function's midpoint
    	'''
    	return L / (1 + np.exp(-k * (x - x_m)))

    def decrement_epsilon(self):
        self.epsilon = (- self.logistic(self.epsilon, 1.0, 1/4, (1/3) * self.iters) + 1) * self.eps_dec \
            if self.epsilon > self.eps_end else self.eps_end

    def update_Q(self, state, action, reward, next_state):
        optimal_action = self.optimal_action(next_state)

        self.Q[(state, action)] = \
            self.Q[(state, action)] + self.lr * \
            self.gamma * self.Q[(next_state, optimal_action)] - \
            self.Q[(state, action)]

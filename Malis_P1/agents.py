"""Agents for Fishing Season (MALIS, Project 1).

This file is a skeleton. Write all your agents in it.

Do NOT modify the classes Agent and ContextualAgent, nor the function play():
they define how every agent interacts with the simulator, and they are used
to run your agents during grading.

How an agent works
------------------
For each season, play(agent, bay) calls:

    agent.reset()                     once, before the first day
    spot = agent.choose()             every day: return a spot, 0 to 4
    agent.update(spot, catch)         every day: the catch you observed

An agent only learns through update(). It must not look at the bay itself.

Contextual agents (Part IV, optional) follow the same pattern, but receive the
morning conditions x (numpy array [wind_kmh, waves_m, tide_m]):

    spot = agent.choose(x)
    agent.update(x, spot, catch)

Example
-------
    from fishing import FishingBay
    from agents import RandomAgent, play

    bay = FishingBay(season=1, seed=0)
    total = play(RandomAgent(seed=0), bay)
"""
import numpy as np

from fishing import ContextualBay, FishingBay

# ---------------------------------------------------------------------------
# Provided: do not modify
# ---------------------------------------------------------------------------


class Agent:
    """Base class for the agents of Parts II and III.

    Parameters
    ----------
    n_spots : int
        Number of fishing spots (5 in the game).
    seed : int or None
        Seed of the agent's own random generator, self.rng. Use self.rng for
        any random decision, so that results can be reproduced.
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self):
        """Return the spot to fish today, an int between 0 and n_spots - 1."""
        raise NotImplementedError

    def update(self, spot, catch):
        """Called after each day with the spot chosen and the catch observed (kg)."""


class ContextualAgent:
    """Base class for the agents of Part IV (contextual bay).

    Same as Agent, except that choose() and update() also receive x, the
    conditions of the day: numpy array [wind_kmh, waves_m, tide_m].
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self, x):
        """Return the spot to fish today given the conditions x."""
        raise NotImplementedError

    def update(self, x, spot, catch):
        """Called after each day with the conditions, the spot chosen and the catch observed."""


def play(agent, bay):
    """Play one full season of bay with agent. Returns the total catch (kg).

    Works with FishingBay and Agent, or with ContextualBay and ContextualAgent.
    The bay is reset first, so the same bay can be played by several agents.
    """
    contextual = isinstance(bay, ContextualBay)
    if contextual and not isinstance(agent, ContextualAgent):
        raise TypeError("a ContextualBay must be played by a ContextualAgent")
    if not contextual and not isinstance(agent, Agent):
        raise TypeError("a FishingBay must be played by an Agent")
    bay.reset()
    agent.reset()
    while not bay.done:
        if contextual:
            x = bay.observe_array()
            spot = int(agent.choose(x))
            catch = bay.fish(spot)
            agent.update(x, spot, catch)
        else:
            spot = int(agent.choose())
            catch = bay.fish(spot)
            agent.update(spot, catch)
    return bay.total


# ---------------------------------------------------------------------------
# Provided example agents (baselines)
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """Chooses a spot uniformly at random every day. Learns nothing."""

    def choose(self):
        return int(self.rng.integers(self.n_spots))


class RandomContextualAgent(ContextualAgent):
    """Ignores the conditions and chooses a spot uniformly at random every day."""

    def choose(self, x):
        return int(self.rng.integers(self.n_spots))


# ---------------------------------------------------------------------------
# Your agents
# ---------------------------------------------------------------------------


class HumanRuleAgent(Agent):
    """Task 4: the best human strategy identified in Part II, written as a rule.

    Describe the rule here, precisely enough that someone could apply it by hand.
    Add constructor parameters if your rule has any (call super().__init__ first).
    """
    
    """
    Threshold strategy:
    Stay at the current spot.
    Switch after two consecutive catches below 10 kg.
    """

    def __init__(self, n_spots=5, seed=None, threshold=10):
        super().__init__(n_spots, seed)
        self.threshold = threshold

    def reset(self):
        self.current_spot = 0
        self.low_count = 0

    def choose(self):
        return self.current_spot

    def update(self, spot, catch):
        if catch < self.threshold:
            self.low_count += 1
        else:
            self.low_count = 0

        if self.low_count >= 2:
            self.current_spot = (self.current_spot + 1) % self.n_spots
            self.low_count = 0


    """def reset(self):
        raise NotImplementedError("Task 4")

    def choose(self):

        raise NotImplementedError("Task 4")

    def update(self, spot, catch):
        raise NotImplementedError("Task 4")"""


# Task 4: add below at least two agents implementing methods from the reading
# (Sutton and Barto, Chapter 2), as subclasses of Agent. Document each class:
# what it keeps in memory, how it chooses, what its parameters mean.
class EpsilonGreedyAgent(Agent):
    """
    Epsilon-greedy agent with sample-average estimates.

    epsilon: probability of choosing a random spot.
    counts: number of visits to each spot.
    values: estimated average catch of each spot.
    """

    def __init__(self, n_spots=5, seed=None, epsilon=0.1):
        super().__init__(n_spots, seed)
        self.epsilon = epsilon

    def reset(self):
        self.counts = np.zeros(self.n_spots, dtype=int)
        self.values = np.zeros(self.n_spots)

    def choose(self):
        # Exploration
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_spots))

        # Exploitation
        best = np.flatnonzero(
            self.values == np.max(self.values)
        )
        return int(self.rng.choice(best))

    def update(self, spot, catch):
        self.counts[spot] += 1

        n = self.counts[spot]
        old_value = self.values[spot]

        self.values[spot] = old_value + (
            catch - old_value
        ) / n

class UCBAgent(Agent):
    """
    Upper Confidence Bound (UCB) agent.

    c: controls exploration.
    counts: number of visits to each spot.
    values: estimated average catch for each spot.
    """

    def __init__(self, n_spots=5, seed=None, c=2.0):
        super().__init__(n_spots, seed)
        self.c = c

    def reset(self):
        self.counts = np.zeros(self.n_spots, dtype=int)
        self.values = np.zeros(self.n_spots)
        self.t = 0

    def choose(self):
        # First try each spot at least once
        unexplored = np.flatnonzero(self.counts == 0)

        if len(unexplored) > 0:
            return int(self.rng.choice(unexplored))

        # Calculate UCB score for each spot
        bonus = self.c * np.sqrt(
            np.log(self.t) / self.counts
        )

        scores = self.values + bonus

        best = np.flatnonzero(scores == np.max(scores))
        return int(self.rng.choice(best))

    def update(self, spot, catch):
        self.t += 1
        self.counts[spot] += 1

        n = self.counts[spot]

        self.values[spot] += (
            catch - self.values[spot]
        ) / n

# Task 6 (optional): add your contextual agents below, as subclasses of
# ContextualAgent.

"_________________________________________________________________________________________"
from fishing import FishingBay
#from agents import RandomAgent, play

'''bay = FishingBay(season=1, seed=0)
total = play(RandomAgent(seed=0), bay)
print("Total: ", total, "Best possible: ", bay.best_possible())
agent.reset()
spot = agent.choose()
agent.update(spot, catch)'''

human_ag= HumanRuleAgent(seed=0)
for i in range(1,3+1):
    bay = FishingBay(season=i, seed=0)
    total=play(human_ag, bay)

    print("Season: ", i, "\nTotal: ", total, "Best possible: ", bay.best_possible(), "\n")
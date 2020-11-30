import numpy as np
import random


class Agent:
    actions = np.array([
        [ 0,  0], # stay
        [-1,  0], # up
        [ 1,  0], # down
        [ 0, -1], # left
        [ 0,  1]  # right
    ])


    def __init__(self, id, env):
        self.id = id
        self.environment = env
        self.Q = np.zeros((self.environment.height, self.environment.width, len(Agent.actions)))
        self.step = 1


    def set_position(self, pos):
        self.position = pos


    def take_action(self, action_id):
        self.step += 1
        if self.check_action_availability(action_id):
            self.environment.grid[self.position[0]][self.position[1]] = 0.0
            self.position += Agent.actions[action_id]
            self.environment.grid[self.position[0]][self.position[1]] = self.id + 1
            return True
        return False


    def epsilon(self):
        return 1.0 / self.step


    def choose_action(self):
        eps = self.epsilon()
        if random.random() < eps:
            # exploration
            action_id = [x for x in range(len(Agent.actions))]
            random.shuffle(action_id)
            for act in action_id:
                if self.check_action_availability(act):
                    return act
        else:
            # exploitation
            q = self.Q[self.position[0]][self.position[1]].copy()
            q_sorted = np.flip(np.sort(q))
            for q_v in q_sorted:
                act = np.random.choice(np.flatnonzero(np.isclose(q, q_v)))
                if self.check_action_availability(act):
                    return act
        return 0


    def check_action_availability(self, action_id):
        if action_id == 0:
            return True
        dest = self.position + Agent.actions[action_id]
        if dest[0] >= 0 and dest[0] < self.environment.height and dest[1] >= 0 and dest[1] < self.environment.width:
            return self.environment.grid[dest[0]][dest[1]] < 1.0
        return False


    def available_actions(self):
        actions = []
        for i in range(len(Agent.actions)):
            if self.check_action_availability(i):
                actions.append(i)
        return actions


    def max_q(self):
        q = self.Q[self.position[0]][self.position[1]][0]
        actions = self.available_actions()
        for act in actions:
            q = max(q, self.Q[self.position[0]][self.position[1]][act])
        return q

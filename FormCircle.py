from Agent import *
import math
import random


class FormCircle:
    def __init__(self, width, height, radius, agent_num):
        self.width = width
        self.height = height
        self.radius = radius
        self.agent_num = agent_num
        self.agents = [Agent(i, self) for i in range(agent_num)]
        self.grid = np.zeros((height, width))


    def init_agents(self):
        pos = random.sample(range(self.width * self.height), self.agent_num)
        pos_x = [(p // self.width) for p in pos]
        pos_y = [(p % self.width) for p in pos]
        for i in range(self.agent_num):
            self.agents[i].set_position(np.array([pos_x[i], pos_y[i]]))
            self.grid[pos_x[i]][pos_y[i]] = i + 1


    def center(self):
        x = np.stack([agent.position[0] for agent in self.agents])
        y = np.stack([agent.position[1] for agent in self.agents])
        return np.array([np.mean(x), np.mean(y)])


    def target_center(self):
        c = self.center()
        x = np.clip(c[0], self.radius, self.height - self.radius)
        y = np.clip(c[1], self.radius, self.width - self.radius)
        return np.array([x, y])


    def agent_displacements(self, base_pos):
        return np.stack([agent.position - base_pos for agent in self.agents])


    def distance_reward(self, agent_id):
        d = self.agents[agent_id].position - self.target_center()
        dist = math.hypot(d[0], d[1])
        return abs(dist - self.radius)


    def uniformity_reward(self):
        d = self.agent_displacements(self.target_center())
        dx = np.stack([v[0] for v in d])
        dy = np.stack([v[1] for v in d])

        angles = np.arctan2(dy, dx) + np.pi
        angles = np.rec.fromarrays([angles, np.arange(self.agent_num)])
        angles.sort()

        angles_diff = np.diff(angles.f0)
        angles_diff = np.append(angles_diff, 2 * np.pi - np.sum(angles_diff))

        target_angle = 2 * np.pi / self.agent_num
        angles_dev = np.abs(angles_diff - target_angle)
        # append agent id
        angles_dev = np.rec.fromarrays([angles_dev, angles.f1, np.roll(angles.f1, -1)])

        rewards = np.zeros(self.agent_num)
        for dev in angles_dev:
            rewards[dev[1]] += dev[0]
            rewards[dev[2]] += dev[0]

        return rewards


    def reward(self, agent_id):
        u = self.uniformity_reward()[agent_id]
        d = self.distance_reward(agent_id)
        return -(u ** (1 + d / math.hypot(self.width, self.height)))

from FormCircle import *


# environment parameters
GRID_WIDTH = 20
GRID_HEIGHT = 20
CIRCLE_RADIUS = 8
AGENT_NUM = 10


# training parameters
ALPHA = 0.05
GAMMA = 0.9
MAX_EPISODE = 1000
EPISODE_MAX_STEP = 100


def train_episode(model):
    env = FormCircle(GRID_WIDTH, GRID_HEIGHT, CIRCLE_RADIUS, AGENT_NUM)
    env.init_agents()
    env.find_target_center()

    # load q table from model
    for i in range(AGENT_NUM):
        env.agents[i].Q = model[i]

    for current_step in range(EPISODE_MAX_STEP):
        print('>> [Step: {}/{}]'.format(current_step + 1, EPISODE_MAX_STEP), end='\r')
        for agent in env.agents:
            action_id = agent.choose_action()
            original_position = agent.position.copy()
            agent.take_action(action_id)
            agent.Q[original_position[0]][original_position[1]][action_id] += ALPHA * (env.reward(agent.id) + GAMMA * agent.max_q() - agent.Q[original_position[0]][original_position[1]][action_id])

    # return model
    return np.stack([agent.Q for agent in env.agents])


if __name__ == '__main__':
    model = np.zeros((AGENT_NUM, GRID_HEIGHT, GRID_WIDTH, 5))
    for current_episode in range(MAX_EPISODE):
        print('\n[Episode: {}/{}]'.format(current_episode + 1, MAX_EPISODE))
        model = train_episode(model)
        print(model)

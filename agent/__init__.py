try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import random
import os
import math
import collections
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 5000
batch_size    = 32
max_episodes  = 5000
t_max         = 600
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'reward_shape', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):

        self.buffer = []
        self.buffer_limit = buffer_limit
        self.index = 0

    def push(self, transition):
        if(self.index >= self.buffer_limit):
            self.index = 0

        self.buffer.insert(self.index, transition)
        self.index = self.index + 1

    def sample(self, batch_size):

        sample = random.sample(self.buffer, batch_size)

        states = torch.tensor([s.state for s in sample if s is not None]).float().to(device)
        actions = torch.tensor([s.action for s in sample if s is not None]).to(device)
        rewards = torch.tensor([s.reward for s in sample if s is not None]).float().to(device)
        rewards_shape = torch.tensor([s.reward_shape for s in sample if s is not None]).float().to(device)
        next_states = torch.tensor([s.next_state for s in sample if s is not None]).float().to(device)
        dones = torch.tensor([int(s.done[0]) for s in sample if s is not None]).float().to(device)

        return states, actions, rewards, rewards_shape, next_states, dones

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)

class Base(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)

class ExampleAgent(Base):
    '''
    An example agent that just output a random action.

    def __init__(self, *args, **kwargs):

        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.

        For example, you might want to load the appropriate neural networks weight
        in this method.

        super().__init__()
        test_case_id = kwargs.get('test_case_id')

        # Uncomment to help debugging
        print('>>> __INIT__ >>>')
        print('test_case_id:', test_case_id)
        '''

    def initialize(self, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        '''
        fast_downward_path  = kwargs.get('fast_downward_path')
        agent_speed_range   = kwargs.get('agent_speed_range')
        gamma               = kwargs.get('gamma')
        '''
        # Uncomment to help debugging
        print('>>> INITIALIZE >>>')
        print('fast_downward_path:', fast_downward_path)
        print('agent_speed_range:', agent_speed_range)
        print('gamma:', gamma)
        '''

    def step(self, state, epsilon=0.0):
        '''
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.

        Input:
        * `state`:  tensor of dimension `[channel, height, width]`, with
                    `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        '''
        '''
        # Uncomment to help debugging
        print('>>> STEP >>>')
        print('state:', state)
        '''

        #print('state:' , state)

        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        qVal = self.forward(state)
        max, index = qVal.max(1)
        best_action = index.item()

        if np.random.random() <= epsilon:
            randomAct = random.randrange(self.num_actions)
            return randomAct
        else :
            return best_action

    def update(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.

        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with
                   `channel=[cars, agent, finish_position, occupancy_trails]`
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependant to its past.
        '''
        state       = kwargs.get('state')
        action      = kwargs.get('action')
        reward      = kwargs.get('reward')
        next_state  = kwargs.get('next_state')
        done        = kwargs.get('done')
        info        = kwargs.get('info')
        '''
        # Uncomment to help debugging
        print('>>> UPDATE >>>')
        print('state:', state)
        print('action:', action)
        print('reward:', reward)
        print('next_state:', next_state)
        print('done:', done)
        print('info:', info)
        '''

class DQN(ExampleAgent):
    def construct(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

class ConvDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        super().construct()

def compute_loss(model, target, states, actions, rewards, rewards_shape, next_states, dones):

    target_val = target.forward(next_states).max(1)[0].unsqueeze(1)

    terminal_states = dones.unsqueeze(1)

    target_val = rewards + rewards_shape + (target_val *  (1  - terminal_states)) * gamma

    model_val = model.forward(states)
    model_val = torch.gather(model_val, 1, actions)

    loss_funct = nn.SmoothL1Loss()
    loss = loss_funct(model_val, target_val)

    return loss

def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def train(model_class, env):
    # Initialize model and target network
    model = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            action = model.step(state, epsilon)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            maximum1= state[1].max(1)
            #print('max1:' , maximum1)
            index_y = np.where(maximum1==1)
            y = index_y[0].item()
            maximum0 = state[1].max(0)
            #print('max0:', maximum0)
            index_x = np.where(maximum0==1)
            x = index_x[0].item()

            phi_state = (- x - y)

            max_next1 = next_state[1].max(1)
            index_next_y = np.where(max_next1==1)
            next_y = index_next_y[0].item()

            max_next0 = next_state[1].max(0)
            index_next_x = np.where(max_next0==1)
            next_x = index_next_x[0].item()

            phi_next_state = (-next_x - next_y)

            #phi_state = (-env.state.channel.agent.position.x - env.state.channel.agent.position.y)
            #phi_next_state = (-env.next_state.channel.agent.position.x - env.next_state.channel.agent.position.y)
            reward_shape = gamma * (phi_next_state - phi_state)
            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], [reward_shape], next_state, [done]))

            state = next_state
            episode_rewards += reward

            if done:
                break

        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            #if np.mean(rewards[print_interval:]) < 0.1:
                #print('Bad initialization. Please restart the training.')
                #exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*10:]), len(memory), epsilon*100))
    return model

def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`.
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model

def save_model(model):
    '''
    Save `model` to disk. Location is specified in `model_path`.
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)

def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    agent = get_model()
    return agent


if __name__ == '__main__':
    import sys
    import time
    from env import construct_task2_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        for run in range(runs):
            state = env.reset()
            agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
            agent.initialize(**agent_init)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state,
                    'done': done, 'info': info
                }
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards)/len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards

    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task():
        tcs = [('task_2_tmax50', 50), ('task_2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_task2_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs]
        }

    task = get_task()

    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    args = parser.parse_args()

    if args.train:
        model = train(ConvDQN, construct_task2_env())
        save_model(model)
    else:
        timed_test(task)

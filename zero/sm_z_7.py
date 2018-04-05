import math
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import numpy as np
from numba import jit
from scipy.integrate import odeint
import os

parser = argparse.ArgumentParser(description='sine RL task')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G')
parser.add_argument('--batch_size', type=int, default=256, metavar='N')
parser.add_argument('--eps_decay', type=int, default=800000, metavar='A')
parser.add_argument('--lr', type=float, default=0.01, metavar='B')
parser.add_argument('--mom', type=float, default=0.8, metavar='B')
parser.add_argument('--exp_replay', type=int, default=20000, metavar='C')
parser.add_argument('--episodes', type=int, default=500, metavar='D')
parser.add_argument('--neurons', type=int, default=128, metavar='E')
parser.add_argument('--folder', type=int, default=0, metavar='F')

args = parser.parse_args()
# folder where graphs and output will be saved
os.mkdir(str(args.folder))

# file where all the data is logged
f = open('{}/log.txt'.format(args.folder), 'w+')

# save all hyperparameters
f.write("HYPERPARAMETERS: \n\n")
for arg in vars(args):
    f.write('{}: {} \n'.format(arg, getattr(args, arg)))
f.write("\n")
f.write("TRAINING: \n\n")
f.flush

use_GUI = False
use_cuda = True

if use_GUI:
    from matplotlib import pyplot as plt


#
# STEPPER MOTOR ENVIRONMENT
#

# constants
Lmean = 0.0020
DeltaL = 0.0004  # 4e-4
CT = 0.2036  # koppelconstante = rotor flux
b = 0.0199  # demping
J = 0.000062758  # inertie
p = 50  # poolparen
La = 0.0020
Lb = 0.0020
Ke = 0.2030
Ra = 0.4500
Rb = 0.4500
Ts = 0.000001
microsteps = 16

# input signals

load_torque = 0.266


# load torque
def T_last(t):
    return load_torque


speed_level = 266


# gewenst snelheids traject
def speed(t):
    return speed_level


current = 4.7


# max stroom amplitude
def current_level(t):
    return current


# help variables
fout_a_g = 0
fout_b_g = 0
fout_a_prev = 0
fout_b_prev = 0
t_old = 0
delta_real = 0
u_a_g = 0
u_b_g = 0
T_em = 0


@jit
def hys(fout, fout_prev):
    du = (fout - fout_prev)
    acc = 40
    if du >= 0:
        u_a_1_1 = (1.0 / (1 + 2.718 ** (-(fout - 0.05) * acc)))
        u_a_1_2 = (1.0 / (1 + 2.718 ** (-fout * acc))) - 1
        return 24 * (u_a_1_1 + u_a_1_2)
    elif du < 0:
        u_a_2_1 = (1.0 / (1 + 2.718 ** (-(fout + 0.05) * acc))) - 1
        u_a_2_2 = (1.0 / (1 + 2.718 ** (-fout * acc)))
        return 24 * (u_a_2_1 + u_a_2_2)


def sm_model(z, t):
    theta = z[0]
    d_theta = z[1]
    i_a = z[2]
    i_b = z[3]
    speed_s = z[4]
    theta_tr = z[5]

    # bepalen van de gewenste hoek van de stroom vector
    beta_g = theta_tr * p
    # gekwantiseerde gewenste beta
    beta_g_quant = (round(beta_g * microsteps * 2 / math.pi) * math.pi) / (2 * microsteps)

    # bepalen van de fase stroomvectoren
    i_b_g = current_level(t) * math.sin(beta_g_quant)
    i_a_g = current_level(t) * math.cos(beta_g_quant)

    # modellering van het koppelgedrag
    Ld_Lq_2 = (-DeltaL * 0.5) + (Lmean * math.sin(p * theta))
    # werkelijke stroom amplitude
    I_s = math.pow(math.pow(i_a, 2) + math.pow(i_b, 2), 0.5)
    # werkelijke stroom beta
    beta = math.atan2(i_b, i_a)
    # last hoek
    delta = beta - (p * theta)
    # elektromechanisch koppel
    global T_em
    T_em = (CT * I_s * math.sin(delta)) + (Ld_Lq_2 * math.pow(I_s, 2) * math.sin(2 * delta))

    # berekenen van delta_real
    # change this to -1 when speed is negative, or calculate it when speed sign changes in one simulation
    direction = 1
    global delta_real
    # scale delta between -pi and pi
    delta_real = ((((delta * direction) + math.pi) % (math.pi * 2)) - math.pi) * (360 / (math.pi * 2))

    # modellering van de elektrische eigenschappen
    # tegen emk fase a formule gebruikt in simulink model
    e_a = -Ke * math.sin(p * theta) * d_theta
    # tegen emk fase b formule gebruikt in simulink model
    e_b = Ke * math.cos(p * theta) * d_theta

    # fout tussen de gewenste en de werkelijke stroom in beide fasen
    fout_a = i_a_g - i_a
    fout_b = i_b_g - i_b

    global fout_a_g
    fout_a_g = fout_a

    global fout_b_g
    fout_b_g = fout_b

    result = [[], [], [], [], [], []]
    result[0] = d_theta  # d_theta
    result[1] = (T_em - T_last(t) - (b * d_theta)) / J  # d2_theta
    result[2] = (u_a_g - (Ra * i_a) - e_a) / La  # d_i_a
    result[3] = (u_b_g - (Rb * i_b) - e_b) / Lb  # d_i_b
    result[4] = 100.0 * (speed(t) - speed_s)  # d_speed_s
    result[5] = 2 * math.pi * speed_s / 60  # d_theta_tr

    return result


class StepperEnv:
    def __init__(self):
        # range of different drive parameters
        self.current_max = 4.7
        self.current_min = 3.0

        self.speed_max = 300
        self.speed_min = 200

        self.load_torque_max = 0.3
        self.load_torque_min = 0.2

        self.time_step = 0.00002
        self.duration = 0.2

        self.delta_max = 180  # 99.7078336473
        self.delta_min = 0  # 81.9967737128

        self.t = 0
        self.step_counter = 0
        self.z0 = [0, 0, 0, 0, 0, 0]

        self.time_arr = []
        self.omega_arr = []
        self.delta_arr = []
        self.T_arr = []
        self.current_arr = []

        self.omega_arr.append(0)
        self.delta_arr.append(0)
        self.T_arr.append(0)
        self.current_arr.append(0)
        self.time_arr.append(self.t)

    # @staticmethod
    def get_norm(self, minm, maxm, val):
        if minm == maxm:
            return val - minm
        else:
            return (val - minm) / (maxm - minm)

    def step(self):
        x = np.arange(self.t, self.t + self.time_step, self.time_step / 1.000000000001)

        global u_a_g, u_b_g
        u_a_g = hys(fout_a_g, fout_a_prev)
        u_b_g = hys(fout_b_g, fout_b_prev)

        y = odeint(sm_model, self.z0, x)
        self.z0 = y[1, :]

        global fout_a_prev, fout_b_prev
        fout_a_prev = fout_a_g
        fout_b_prev = fout_b_g

        self.time_arr.append(self.t)
        self.omega_arr.append(y[1][1])
        self.delta_arr.append(delta_real)
        self.T_arr.append(T_em)
        self.current_arr.append(current)

        self.step_counter += 1
        self.t += self.time_step

    def reset(self):
        self.t = 0
        self.step_counter = 0
        self.z0 = [0, 0, 0, 0, 0, 0]

        self.time_arr = []
        self.omega_arr = []
        self.delta_arr = []
        self.T_arr = []
        self.current_arr = []

        self.omega_arr.append(0)
        self.delta_arr.append(0)
        self.T_arr.append(0)
        self.current_arr.append(0)
        self.time_arr.append(self.t)

        global current
        current = self.current_max

    def get_state(self):
        load_norm = self.get_norm(self.load_torque_min, self.load_torque_max, load_torque)
        speed_norm = self.get_norm(self.speed_min, self.speed_max, speed_level)
        delta_norm = self.get_norm(self.delta_min, self.delta_max, delta_real)
        # in principe ook de maximale stroom

        return FloatTensor([float(speed_norm), float(load_norm), float(delta_norm)]).view(1, 3)  # Floattensor size 1x3

    def act(self, a):
        global current
        # current += self.actions[a]
        if a == 0:
            if current > self.current_min:
                current -= 0.005
        elif a == 1:
            if current > self.current_min:
                current -= 0.0005
        elif a == 2:
            if current > self.current_min:
                current -= 0.0001
        elif a == 3:
            if current < self.current_max:
                current += 0.0001
        elif a == 4:
            if current < self.current_max:
                current += 0.0005
        elif a == 5:
            if current < self.current_max:
                current += 0.005

    def get_reward(self):
        # dense reward
        if delta_real < 150:
            return self.current_max - current
            # alternitave reward function return T_em/current
        else:
            return -1

# create an instace of the stepper motor environment
stepper = StepperEnv()

# if gpu is to be used
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

######################################################################
# Replay Memory
# -------------

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# Q-network
# ^^^^^^^^^


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        N = args.neurons
        self.fc1 = nn.Linear(3, N, True)
        self.fc2 = nn.Linear(N, N, True)
        self.fc3 = nn.Linear(N, N, True)
        self.fc4 = nn.Linear(N, 7, True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


######################################################################
# Training
# --------
#

BATCH_SIZE = args.batch_size
GAMMA = args.gamma
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = args.eps_decay

model = DQN()

if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
memory = ReplayMemory(args.exp_replay)

steps_done = 0

eps_threshold = 0


def select_action(state):
    global steps_done
    sample = random.random()
    global eps_threshold
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(7)]])


######################################################################
# Training loop
# ^^^^^^^^^^^^^


last_sync = 0


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    global running_loss
    global l_count
    running_loss += float(loss[0])
    l_count += 1
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` variable. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes.

num_episodes = args.episodes

loss_arr = []
reward_arr = []
delta_arr = []
time_arr = []
eps_arr = []
running_reward = 0
running_loss = 0
r_count = 0
l_count = 0

for i_episode in range(num_episodes):
    # Initialize the environment and state
    stepper.reset()

    state = stepper.get_state()

    running_reward = 0
    running_loss = 0
    r_count = 0
    l_count = 0

    while stepper.t < stepper.duration:
        # Select and perform an action
        if stepper.t > 0.05:
            action_index = select_action(state)
            stepper.act(action_index[0, 0])

        stepper.step()

        reward = stepper.get_reward()
        running_reward += reward
        r_count += 1
        reward = Tensor([reward])

        # Observe new state
        next_state = stepper.get_state()

        if stepper.t > 0.05 + stepper.time_step:
            # Store the transition in memory
            memory.push(state, action_index, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()

        if float(reward[0]) == -1 and stepper.t > 0.05:
            break

    loss_arr.append(running_loss / l_count)
    reward_arr.append(running_reward / r_count)
    time_arr.append(i_episode)
    eps_arr.append(eps_threshold)

    if i_episode % 10 == 0:
        if use_GUI:
            plt.plot(stepper.time_arr, stepper.current_arr)
            plt.savefig('{}/c{}.png'.format(args.folder, i_episode))
            plt.close()
            plt.plot(stepper.time_arr, stepper.delta_arr)
            plt.savefig('{}/d{}.png'.format(args.folder, i_episode))
            plt.close()
        else:
            np.savetxt('{}/t{}.txt'.format(args.folder, i_episode), stepper.time_arr, delimiter=',')
            np.savetxt('{}/c{}.txt'.format(args.folder, i_episode), stepper.current_arr, delimiter=',')
            np.savetxt('{}/d{}.txt'.format(args.folder, i_episode), stepper.delta_arr, delimiter=',')

    # log the output
    f.write("episode: {}, loss: {}, reward: {}, eps: {} \n"
            .format(i_episode, running_loss / l_count, running_reward / r_count, eps_threshold))
    f.flush()

f.write("Complete!")
f.close()

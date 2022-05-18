############### RL for modeling mice decision making ###################
#################### Sanjukta Krishnagopal #############################
######################## sanju33@gmail.com #############################
############################ April 2022 ################################
import gym
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.signal import savgol_filter
import random
from gym import spaces
from matplotlib.pyplot import *
import torch
from torch import nn
from torch import optim
random.seed(42)
from matplotlib.pyplot import *
from numpy import *
import itertools
import random
import csv
from torch import distributions
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.ndimage import zoom
from itertools import chain
import scipy.interpolate as interp
from scipy.ndimage import zoom
from pathlib import Path


def discount_rewards(rewards, gamma=gamma):
    r = np.array([gamma**i * rewards[i] 
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r

#input dictionary t_diff from data. keys: discrete contrasts, values: a list with elements corresponding to a trial containing time passed between consecutive wheel movements (as a list from the IBL dataset)

block_learn_time = []
zero_gap = []
neuron_activity_diff = []
lr_frac_cor_zero = {}
#Hyperparameters
alpha = 0.001 #learning rate
naming = 15
NUM_EPISODES = 1000
counts = 1000 #how many to plot
gamma = 1.0 #discount factor

class agent:
    def __init__(self):

        self.action_space = spaces.Discrete(3)
        self.t = 1
        self.dt = 0.1

        self.attention=True

        self.state = None
        self.c =  [random.uniform(0,1) for i in range(50)]
        self.nu = 0.001 #update of average reward parameter
        self.noise_inattention_perc = 0.05


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, contrast, rho_old):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        (e, x, state_var,state_mean, t) = self.state
        x += (action-1)*self.dt

        if self.t%1 ==0:
            sigma_kappa = 0.1
            kappa_mean = np.random.normal(contrast, sigma_kappa)
            sigma_noise = 0.05
            random_noise = np.random.normal(0,sigma_noise)
            if(self.attention == True):
                state_mean = state_mean + state_var*(kappa_mean - state_mean)/(state_var + sigma_kappa**2)
                state_var=state_var - (state_var*sigma_kappa**2)/(state_var+sigma_kappa**2)
            else:
                state_mean = state_mean + random_noise
                state_var = state_var + sigma_noise**2 


        self.state = (e, x, state_var, state_mean, self.t/100)

        done = (done_l, done_r) = (x <= -5.0, x >= 5.0)

        if not (done[0] or done[1]):
            reward =0
            rho_new = rho_old
        elif contrast > 0:
            if done[1]:
                R=10
                rho_new = rho_old +self.nu*(R-rho_old*self.t)
                reward=R-rho_new*self.t
            elif done[0]:
                R=-50
                rho_new = rho_old +self.nu*(R-rho_old*self.t)
                reward=R-rho_new*self.t
        elif contrast < 0:
            if done[0]:
                R=10
                rho_new = rho_old +self.nu*(R-rho_old*self.t)
                reward=R-rho_new*self.t
            elif done[1]:
                R=-50
                rho_new = rho_old +self.nu*(R-rho_old*self.t)
                reward=R-rho_new*self.t

        elif contrast == 0 and (done[0] or done[1]):
            R=10
            rho_new = rho_old +self.nu*(R-rho_old*self.t)
            reward=R-rho_new*self.t
        self.t+=1

        if self.t > 500:
            done = (True, True)
            R=-50
            rho_new = rho_old +self.nu*(R-rho_old*self.t)
            reward=R-rho_new*self.t

        return np.array(self.state), reward, done, rho_new, {}

    def reset(self):
        self.state = (0,0,1,0,0)
        self.t = 0
        e=0
        return np.array(self.state)

        if self.state is None:
            return None

episode_rewards = []
class policy_estimator(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_inputs = len(env.reset())
        self.n_outputs = env.action_space.n
        # Define network
        self.l1 = nn.Linear(self.n_inputs-3, 64)
        self.l2 = nn.Linear(64, self.n_outputs)

    def forward(self, state):
        z = F.relu(self.l1(torch.FloatTensor(np.squeeze(state[:,2:4]))))
        action_probs = nn.Softmax(dim=-1)(self.l2(z))
        return z, action_probs


##########################################################################

# Create gym and seed numpy
env = agent()
nA = env.action_space.n
np.random.seed(1)


# Main loop
contrast_options = [-1.0, -0.25,-0.125,-0.0625,0.0, 0.0625,0.125,0.25, 1.0]
end_pos=[]
contr=[]
policy_estimator = policy_estimator(env)
optimizer = optim.Adam(policy_estimator.parameters(), lr=alpha)


full_episodes=[]
total_rewards = []
batch_rewards = []
batch_actions = []
batch_states = []
batch_counter = 1
batch_size = 1

all_states =[]
timestamps_RL = []
timestamps_mouse_choice = []
pos_mouse_choice = []
right_block = False
left_block = False
block = []

left_l1_weights=[]
left_l2_weights=[]
left_hidden_neurons = []
right_l1_weights=[]
right_l2_weights=[]
right_hidden_neurons = []

mean_attention_times = {}
mean_spaceout_times = {}

scale = 0.0014 #delta t

all_tdiff=np.array(all_tdiff)
th = np.percentile(all_tdiff,99)

blocks = []
block_switch_time = random.randint(30,50)
block=0.5

neuron_activity_rightblock = dict(zip(np.array(contrast_options), [[],[],[],[],[],[],[],[],[]])) 
neuron_activity_leftblock = dict(zip(np.array(contrast_options), [[],[],[],[],[],[],[],[],[]]))


blocktime=0
left_block_neuron_activity= []
right_block_neuron_activity= []
rho_old = 0.001
for episode in range(NUM_EPISODES):
    done = (False, False)
    state = env.reset()[None,:]

    actions = []
    states = []
    rewards = []
    inatt_states =[]
    # Keep track of game score
    score = 0
    if blocktime==1:#block_switch_time: #run no block for block_switch_time then switch to right block
        block=0.8
    if blocktime % block_switch_time ==0:
        block = abs(1-block)
    if round(block,2) ==0.5:
         contrast = np.random.choice(contrast_options)
    if round(block,2) ==0.8:
        contrast = np.random.choice(contrast_options, 1,p=[0.045,0.045,0.045,0.045,0.1,0.18,0.18,0.18,0.18])[0]
    elif round(block,2) ==0.2: #leftblock
        contrast = np.random.choice(contrast_options, 1,p=[0.18,0.18,0.18,0.18,0.1,0.045,0.045,0.045,0.045])[0]

    #store neuronal activity within each block:
    state_neur_act = np.array((0,0,0,0.5,0))[None,:] #mean 0, stdev 0.5 internal state
    z, _= policy_estimator(state_neur_act)
    if blocktime % (block_switch_time-1) ==0:
        if round(block,2) ==0.8:
            right_block_neuron_activity.append(np.array(z.detach()))
        if round(block,2) ==0.2:
            left_block_neuron_activity.append(np.array(z.detach()))

    blocktime+=1

    t_count = 0
    contr_pos = np.where(all_contrast_m==contrast)[0]

    rand_ = np.random.randint(len(t_diff[contrast]))
    tdiff_choose = t_diff[contrast][rand_]
    mouse_pos_choose =  position_mouse_tdiff[contrast][rand_]
    RL_realtime = []

    while not (done[0] or done[1]):
            if len(tdiff_choose)> t_count:
                if tdiff_choose[t_count]>th:
                    env.attention=False
                else:
                    env.attention=True

            else:
                env.attention=True
            #env.attention=True
            if len(RL_realtime)==0:
                RL_realtime.append(tdiff_choose[t_count])
            elif len(tdiff_choose)> t_count:
                RL_realtime.append(tdiff_choose[t_count] + RL_realtime[-1])
            else:
                RL_realtime.append(np.min(tdiff_choose) + RL_realtime[-1])
            z, action_probs = policy_estimator(state)
            if round(block,2) ==0.8:
                neuron_activity_rightblock[contrast].append(np.array(z.detach()))
            if round(block,2) ==0.2:
                neuron_activity_leftblock[contrast].append(np.array(z.detach()))
            action = np.random.choice(nA,p=action_probs.detach().numpy())
            if(env.attention==False):
                action = 1
            next_state,reward,done,rho_new,_ = env.step(action, contrast, rho_old)
            next_state = next_state[None,:]
            rho_old = rho_new
            #print (next_state[:,2:4])

            inatt_states.append(state[0])
            if env.attention == True:
                states.append(state[0])
                actions.append(action)
                rewards.append(reward)

            score+=reward
            state = next_state

            t_count+=1
            if done[0] or done[1]:
                batch_rewards.extend(discount_rewards(
                rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch/episode is complete, update network
                if batch_counter % batch_size ==0:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    action_tensor = torch.LongTensor(
                       batch_actions)

                    # Calculate loss
                    _, loss_batch = policy_estimator(state_tensor)
                    logprob = torch.squeeze(torch.log(
                        loss_batch))
                    selected_logprobs = reward_tensor *  torch.gather(logprob, 1, action_tensor.unsqueeze(1)).squeeze()
                    loss = -selected_logprobs.mean()
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])

                contr.append(contrast)
                blocks.append(block)
                if done[0] and not done[1]:
                    end_pos.append(-1)
                    #print ('done_break')


                elif done[1] and not done[0]:
                    end_pos.append(1)
                    #print ('done_break')
                else:
                    end_pos.append(0)



    # Append for logging and print
    episode_rewards.append(score)
    all_states.append(inatt_states)
    timestamps_RL.append(np.array(RL_realtime)/scale)
    timestamps_mouse_choice.append(np.array(tdiff_choose))
    pos_mouse_choice.append(np.array(mouse_pos_choose))
    if episode % 10==0: 
        print("Epi " + str(episode) + " Score: " + str(score), ' Time: '+ str(len(actions)) )


#####################plotting###############

#plot sample trials


distinct_cont = np.unique(contr)
contr_time=dict(zip(distinct_cont, [[],[],[],[],[],[],[],[],[]])) 
time_vals = []
mouse_times = [g[-1] for g in timestamp]
RL_times = [shape(g)[0] for g in all_states]
scale = np.min(mouse_times)/np.min(RL_times) #mouse_realtime / RL timesteps
f=0
y_scale = 0.27/5
for c in np.unique(contr):
    s = np.where(contr == c)[0]
    figure()
    for i in s[-30:]:
        contr_time[c].append(timestamps_RL[i][-1])
        plot(np.array(timestamps_RL[i]),np.array(all_states[i])[:,1]*y_scale)
    xlim(0,1000)
    ylim(0.3,-0.3)
    yticks(fontsize=15)
    xticks(fontsize=15)
    plt.title ('Contrast ' + str(c),fontsize=15)
    xlabel('Time(ms)',fontsize=15)
    ylabel('Position',fontsize=15)
    #show()
    savefig('RL_traj'+str(f)+'_.pdf', bbox_inches = 'tight')
    f+=1





####identify %right as a function of contrast:

#right block
nonzero_contrast = np.count_nonzero(contr)
contr_right_rb=dict(zip(distinct_cont, np.zeros(len(distinct_cont))))
contr_total_rb=dict(zip(distinct_cont, np.zeros(len(distinct_cont))))
contr_right_lb=dict(zip(distinct_cont, np.zeros(len(distinct_cont))))
contr_total_lb=dict(zip(distinct_cont, np.zeros(len(distinct_cont))))
contr_right_equal=dict(zip(distinct_cont, np.zeros(len(distinct_cont))))
contr_total_equal=dict(zip(distinct_cont, np.zeros(len(distinct_cont))))
total_rb=0
total_lb=0
for i in arange(len(contr))[-counts:]:
    #right block
    if round(blocks[i],2)==0.8:# or block[i]==100:
        total_rb+=1
        contr_total_rb[contr[i]]+=1
        if end_pos[i]>0:
            contr_right_rb[contr[i]] +=1  #end at right
    #left block
    if round(blocks[i],2)==0.2:# or block[i]==-100:
        total_lb+=1
        contr_total_lb[contr[i]] +=1
        if end_pos[i] > 0 :
            contr_right_lb[contr[i]] +=1
    if round(blocks[i],2)==0.5:
        contr_total_equal[contr[i]] +=1
        if end_pos[i] > 0 :
            contr_right_equal[contr[i]] +=1

#plot %right for the block structure
a1 = np.array(list(contr_right_rb.values()))
a2= np.array(list(contr_total_rb.values()))
a = a1/a2
b1=np.array(list(contr_right_lb.values()))
b2=np.array(list(contr_total_lb.values()))
b = b1/b2
c= (a1[4]+b1[4])/(a2[4]+b2[4])
lr_frac_cor_zero[alpha]=c
print ('right: '+ str(a))
print ('left: '+ str(b))
figure()
ind = np.arange(len(distinct_cont))

p1 = plt.plot(ind, a, label ='80-20')
p2 = plt.plot(ind, b, label ='20-80')
plt.ylabel('Rightward choice (%)', fontsize = 15)
plt.xlabel('Contrast', fontsize = 15)
ylim(0,1)
legend()
plt.xticks(ind, list(contr_right_rb.keys()))
plt.savefig('Percentright_by_contrast_RL.pdf', bbox_inches='tight')
zero_gap.append(abs(a[4]-b[4]))







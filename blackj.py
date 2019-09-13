
import numpy
import gym 
from matplotlib import pyplot
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial


plt.style.use('ggplot')


env = gym.make('Blackjack-v0')
env.action_space, env.observation_space

def sample_policy(observation):
  score, dealer_score, usable_ace = observation
  return 0 if score >= 20 else 1 

def generate_episode(policy,env):
  states, actions, rewards = [],[],[]
  observation = env.reset()
  while True:
    print('observation = ',observation)
    states.append(observation)
    print('states =',states)
    action = sample_policy(observation)
    print('action =',action)
    actions.append(action)
    print('actions =',actions)
    observation,reward,done, info = env.step(action)
    print(observation,reward,done, info)
    print('reward = ',reward)
    rewards.append(reward)
    print('rewards = ',rewards)
    if done:
      break
  print(states,actions,rewards) 
  return states,actions,rewards

def first_visit_mc_prediction(policy,env,n_episodes):
  value_table = defaultdict(float)
  print(value_table)
  N = defaultdict(int)
  print(N) 
  for _ in range(n_episodes):
    states,_,rewards = generate_episode(policy,env)
    returns = 0 

    for t in range(len(states) -1,-1,-1):
      R = rewards[t]
      S = states[t]
      returns += R
      if S not in states[:t]:
        N[S] += 1 
        value_table[S] += (returns-value_table[S])/N[S]
  print('value_table = ',value_table)
  return value_table

value = first_visit_mc_prediction(sample_policy,env,n_episodes=50000)
#print(value)


def plot_blackjack(V,ax1,ax2):
  player_sum = numpy.arange(12,21+1)
  dealer_show = numpy.arange(1,10+1)
  usable_ace = numpy.array([False,True]) 
  state_values = numpy.zeros((len(player_sum),len(dealer_show),len(usable_ace)))

  for i, player in enumerate(player_sum):
    for j, dealer in enumerate(dealer_show):
      for k, ace in enumerate(usable_ace):
        state_values[i,j,k] = V[player,dealer,ace]

  x,y = numpy.meshgrid(player_sum,dealer_show)

  ax1.plot_wireframe(x,y,state_values[:,:,0])
  ax2.plot_wireframe(x,y,state_values[:,:,1])
  for ax in ax1,ax2:
    ax.set_zlim(-1,1) 
    ax.set_ylabel('player sum')
    ax.set_xlabel('dealer showing')
    ax.set_zlabel('state-value')



fig,axes = pyplot.subplots(nrows=2,figsize=(5,8),subplot_kw={'projection':'3d'})
axes[0].set_title('value function without usable ace')
axes[1].set_title('value function with usable ace')
plot_blackjack(value,axes[0],axes[1])






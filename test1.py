
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
  print(env.render())
  env.render()
  print(env.action_space.sample())
  env.step(env.action_space.sample())
  print(env.step(env.action_space.sample()))




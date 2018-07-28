# Reinforcement Learning Papers implementation
Implementation of RL papers, tested in the Cartpole problem. 
This project was done after having watched several lectures and read papers about Reinforcement Learning (RL), an area in which I have a deep interest in. The objective was to take what I had learned and apply it to a small but interesting problem.
 
I decided to program an agent to learn how to solve Cartpole, using the [OpenAI Gym](https://gym.openai.com/docs/) environment, which facilitated the development. I decided to implement several Q-Learning algorithms: [Deep Q-Learning](https://deepmind.com/research/dqn/), [Double Deep Q-Learning](https://arxiv.org/abs/1509.06461) and [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952). Also, I wanted to implement a Policy Gradient solution, so I decided to implement [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), and an Actor-Critic one, so I implemented [A3C](https://arxiv.org/abs/1602.01783).

This was done using [Keras](https://keras.io/) as the framework to program the Neural networks, with Python as the programming language of choice. When doing the Policy/Actor-Critic methods, I started using [Pytorch](https://pytorch.org/) as I needed more control on gradient updates and I realized it was a good opportunity to learn this framework as well.

## Results
![Results in Cartpole for Q-Learning mehhods](graphs/cartpole_results.png)

![Results in Cartpole for Policy methods](graphs/cartpole_results_pg.png)

The graphic above shows the results obtained at Cartpole for each implemented algorithm. The results were obtained using the median of five runs and smoothed with moving average. Each run was finished once the agent averaged more than 200 points of reward in the last 100 episodes.

## Aknowledgements
Other than the linked papers, I have to thank Jaromir Janisch for his great [article](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/), which helped me a lot. I used his Sum Tree implementation, since it was pretty simple. 

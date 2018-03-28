# Deep-Reinforcement-Learning-for-FlappyBird

We trained a Artificial Intelligence model to play FlappyBird with images as inputs. The model receives the game's screen and decides whether the bird should fly or fall. It achieves a higher average performance than human players.

# Demo

[![Demo CountPages alpha](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/Artificial_Intelligence_playing_FlappyBird.gif)](https://www.youtube.com/watch?v=Tf8SVv1nPxM)

# Requirements

You require to install not only the Python dependencies, but also PyGame and GymPLE, PyGame's adaptation for Gym environments.

To install PLE:

	$ git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
	$ cd PyGame-Learning-Environment/
	$ pip install -e .

To install PyGame on OSX:

	$ brew install sdl sdl_ttf sdl_image sdl_mixer portmidi  # brew or use equivalent means
	$ conda install -c https://conda.binstar.org/quasiben pygame  # using Anaconda

On Ubuntu 14.04:

	$ apt-get install -y python-pygame

More configurations and installation details on: http://www.pygame.org/wiki/GettingStarted#Pygame%20Installation



And finally clone and install this package

	$ git clone https://github.com/lusob/gym-ple.git
	$ cd gym-ple/
	$ pip install -e .

# Run a demo

    $ git clone https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird.git
    $ cd Deep-Reinforcement-Learning-for-FlappyBird
    $ pip install -r requirements.txt
    $ python scripts/test_model.py

# Train your own model

    $ python scripts/training_with_preprocessing.py

We trained our model with 1,500,000 iterations (almost 18,000 games). It took 17 hours on a G3 AWS instance (NVIDIA Tesla M60 GPUs). Here is the learning curve we get (average reward as a function of the number of games played):

[![Demo CountPages alpha](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/Learning%20curve.png)](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/paper.pdf)

# Analysis

One can find a paper (pre-print version) that explains our method, why it performs and things one could improve, at the following link: https://www.researchgate.net/profile/Louis_Samuel_Pilcer/publication/324066514_Playing_Flappy_Bird_with_Deep_Reinforcement_Learning/links/5abbc2230f7e9bfc045592df/Playing-Flappy-Bird-with-Deep-Reinforcement-Learning.pdf.

# Environment

At a given time the environment is in a given state (location and direction of the bird, location of pipes...) that translates into a 512x288 pixels colored image. At any time the agent can perform two types of actions:

- a=0: do nothing
- a=1:fly.

These actions can result in negative reward (the bird crashes before the first obstacle) or in positive rewards (the bird passes some obstacles and crashes between the reward and the reward+1 obstacle) at the end of the game. Positive rewards are based on the number of obstacles the bird passes. When the agent performs any action, the environment changes, leading to a new state.

We model this set of states, actions and rewards, as a Markov decision process.

# Deep Q-Learning

We keep in memory a set of former games, as tuples of (*state, action, discounted reward*) with a discount rate 0.99 for experiments.

At any given time, our agent tries to evaluate the reward it will achieve with both actions, using what we call a Deep Q-Network:

[![Demo CountPages alpha](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/Deep%20Q-Network.png)](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/paper.pdf)

In traditional Q-Learning, agents try to evaluate this Q-function based on previous experiments that crossed the same state. As our state-space is extremely large, here we try to *generalize* knowledge we acquired on similar states to evaluate the Q-function on a given state.

Our agent needs to be able to analyze the image and, without any prior knowledge besides interaction with its environment, to learn strategies that enable the bird to pass obstacles. We thus needed a model able to learn features on an image, such as the distance (altitude/longitude) between the bird and the next pipe, and to process these features in order to predict Q-values. We used the following model introduced in Google's paper Playing Atari with Deep Reinforcement Learning (https://arxiv.org/pdf/1312.5602v1.pdf):

[![Demo CountPages alpha](https://media.springernature.com/m685/nature-static/assets/v1/image-assets/nature14236-f1.jpg)](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/paper.pdf)

This network is inspired by others that perform well on Computer Vision problems (http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf, https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf). The architecture helps the agent learn from experience ways to build abstract features on input images (convolutional layers), and to compose these features in order to understand how good the current position is / would be depending on the action it takes (dense layers).

Our agent plays games, and keeps the rewards achieved for every (*state, action*) in memory. Every 10,000 iterations, we take a batch of previous experiences and the reward associated and use them to train the convolutional neural network used for Q-value prediction.


# Image preprocessing

To make training faster, we built a preprocessing method thas consists on erasing the background image and keeping only the bird, pipes and the ground, with binary features. Our idea was that these preprocessed images would make bird detection easier (just have to find a circular zone where pixels are equal to 1) and that it would enable the model to learn easily the distance between the bird and the pipe, the altitude difference, etc.


# Experience replay

In order to stabilize over time the Q−values given by the approximation of the Q−function, we used the technique of experience replay introduced by Deepmind. Experience replay consists of storing a defined number of the last experiences in a replay memory and to randomly use them when running the gradient descent algorithm that trains the neural network. This process of experience replay might reduce oscillations as we train our network not only on recent observations/rewards, but also on data that we randomly sample in the agent’s memory. We train our model on batches of data that might represent many past behaviors.


# The exploration/exploitation dilemma

The neural network being initialized randomly, it initially acts following a random policy and then will be improved by training until it finds a successful strategy. But the first successful strategy isnt necessarily the best one. The question the network should address is : should I exploit the known working strategy or explore other, that may be better strate- gies? In fact, the first strategy is greedy as it sticks with the first effective type of policy it discovers.

In order to avoid this problem of sticking with a non- really-effective strategy, we use ε-greedy exploration: with probability ε choose a random action, otherwise choose the action that maximises the Q−function. In our model, just as in Deepmind’s model, ε value decreases linearly over iterations from 1 (random actions) to 0.1.

This way we ensure to avoid sticking with non-really- effective strategies. However, 


# Improvements



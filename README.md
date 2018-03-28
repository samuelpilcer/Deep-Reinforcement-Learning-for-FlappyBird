# Deep-Reinforcement-Learning-for-FlappyBird

We trained a Deep Reinforcement Learning model to play FlappyBird, using screens as inputs. The model receives the game's screen and decides whether the bird should fly or fall. It achieves a higher average performance than human players.

# Analysis

One can find our report's pre-print at the following link: https://www.researchgate.net/profile/Louis_Samuel_Pilcer/publication/324066514_Playing_Flappy_Bird_with_Deep_Reinforcement_Learning/links/5abbc2230f7e9bfc045592df/Playing-Flappy-Bird-with-Deep-Reinforcement-Learning.pdf

# Environment

At a given time the environment is in a given state (location and direction of the bird, location of pipes...) that translates into a 512x288 pixels colored image. At any time the agent can perform two types of actions:

- a=0: do nothing
- a=1:fly.

These actions can result in negative reward (the bird crashes before the first obstacle) or in positive rewards (the bird passes some obstacles and crashes between the reward and the reward+1 obstacle) at the end of the game. Positive rewards are based on the number of obstacles the bird passes. When the agent performs any action, the environment changes, leading to a new state.

We model this set of states, actions and rewards, as a Markov decision process.

# Deep Q-Learning

We keep in memory a set of former games, as tuples of (*state, action, discounted reward*) with a discount rate 0.99 for experiments.

[![Demo CountPages alpha](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/Deep%20Q-Network.png)](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/paper.pdf)

# Pipeline

We used the model introduced in Google's paper Playing Atari with Deep Reinforcement Learning (https://arxiv.org/pdf/1312.5602v1.pdf) 

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

We trained our model with 1,500,000 iterations (almost 18,000 games). It took 17 hours on a G3 AWS instance (NVIDIA Tesla M60 GPUs). Here is the learning curve we get (reward as a function of the number of games played):

[![Demo CountPages alpha](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/Learning%20curve.png)](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/paper.pdf)

# Demo

[![Demo CountPages alpha](https://github.com/samuelpilcer/Deep-Reinforcement-Learning-for-FlappyBird/blob/master/experiment/Artificial_Intelligence_playing_FlappyBird.gif)](https://www.youtube.com/watch?v=Tf8SVv1nPxM)

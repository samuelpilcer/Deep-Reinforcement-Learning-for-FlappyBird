# Deep-Reinforcement-Learning-for-FlappyBird

We trained a Deep Reinforcement Learning model to play FlappyBird, using screens as inputs. The model receives the game's screen and decides whether the bird should fly or fall. It achieves a higher average performance than human players.

# Deep Q-Learning

We keep in memory former 

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

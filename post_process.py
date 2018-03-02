import matplotlib.pyplot as plt


def test_agent(env, agent, nb_episodes=10):
    env.reset()
    agent.test(env, nb_episodes=nb_episodes, visualize=True)


def plot_reward(history):
    plt.plot(history.history["episode_reward"])
    plt.show()


def plot_steps(history):
    plt.plot(history.history["nb_episode_steps"])
    plt.show()


def save_model(model, model_json_filename, model_weights_filename):
    model_json = model.to_json()
    with open(model_json_filename, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_weights_filename)

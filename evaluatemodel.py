import gym
import tensorflow as tf
import keras.backend as K
from keras.models import model_from_json


def main():
    sess = tf.compat.v1.Session()
    K.set_session(sess)
    env = gym.make("MountainCarContinuous-v0", render_mode='human')

    episodes = 400
    time_steps = 501

    # load json and create model
    json_file = open('./Model architecture and trained weights/Actor_model_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    actor = model_from_json(loaded_model_json)
    actor.load_weights("./ModelWeights/DDPG_actor_model_750.h5")

    run = True
    if run:
        for episode in range(episodes):
            state = env.reset()[0]
            episode_reward = 0

            for time in range(time_steps):
                env.render()
                action = actor.predict(state.reshape((1, 2)))[0]

                # print("deterministic action:",action)
                # print("noisy action:", exploratory_action)

                next_state, reward, terminated, truncated, _ = env.step(action)

                episode_reward += reward
                state = next_state

                if terminated:
                    break
            print("Completed in {} steps.... episode: {}/{}, episode reward: {} "
                  .format(time, episode, episodes, episode_reward))
        env.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import clone

NUM_TEST = 20
NUM_EPISODES = 10000
GPU_USAGE =0.05

def test(env, max_steps, trained_clone):
    """
    Tests the behavioral clone.
    """
    print("testing")
    returns = []
    for episode in range(NUM_TEST):
        obs = env.reset()
        totalr = 0
        for time in range(max_steps):
            act = trained_clone.act(obs)

            obs, r, done, _ = env.step(act)
            totalr += r

            if done:
                break
        returns += [totalr]
        print("iter {}: {}".format(episode, totalr))

    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=10,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_USAGE

    with tf.Session(config=config) as sess:
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit



        resume = args.resume
        name = args.name
        num_rollouts = args.num_rollouts
        if resume: 
            print('loading a basic clone {}'.format(name))
        else:
            print('creating a basic clone {}'.format(name))

        bclone = clone.BasicClone(
            list(env.observation_space.shape),
            list(env.action_space.shape), sess, name, load=resume)
        print('clone ready.')

        if args.test:
            test(env, max_steps, bclone)
        else:
            for i in range(num_rollouts):
                if i % 5 == 0 and i > 0:
                    print('saving model.')
                    bclone.save()
                    print('model saved.')

                print('iter', i)
                obs = env.reset()
                done = False
                steps = 0
                while not done:
                    action = policy_fn(obs[None,:])


                    loss = bclone.perceive(obs, action[0])

                    obs, r, done, _ = env.step(action)
                    steps += 1

                    if args.render:
                        env.render()
                    if steps % 100 == 0:
                        print("{}\t {}".format(steps+i*max_steps, loss))
                    if steps >= max_steps:
                        break

            test(env, max_steps, bclone)


if __name__ == '__main__':
    main()

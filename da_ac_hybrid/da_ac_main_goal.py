import argparse
import os

import gym
import torch
import numpy as np

from agents import da_ac
from common import utils
from common.env_wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from common.goal_wrappers import GoalFlattenedActionWrapper, GoalObservationWrapper


def pad_action(act, act_param):
    params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
    params[act] = act_param
    return (act, params)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate(env, policy, action_parameter_sizes, episodes=100):
    returns = []
    epioside_steps = []

    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            discrete_action, all_parameter_action = policy.select_action(state, eval=True)
            offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
            parameter_action = all_parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
            action = pad_action(discrete_action, parameter_action)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        epioside_steps.append(t)
        returns.append(total_reward)
    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-100:]).mean():.3f} success: {(np.array(returns) == 50.).sum() / len(returns):.3f} epioside_steps: {np.array(epioside_steps[-100:]).mean():.3f}")
    print("---------------------------------------")
    return np.array(returns[-100:]).mean(), np.array(epioside_steps[-100:]).mean(), (
            np.array(returns) == 50.).sum() / len(returns)


def run(args):
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    if args.env == "Goal-v0":
        env = gym.make('Goal-v0')
        env = GoalObservationWrapper(env)
        kickto_weights = np.array([[-0.375, 0.5, 0, 0.0625, 0],
                                   [0, 0, 0.8333333333333333333, 0, 0.111111111111111111111111]])
        shoot_goal_left_weights = np.array([0.857346647646219686, 0])
        shoot_goal_right_weights = np.array([-0.857346647646219686, 0])
        initial_weights = np.zeros((4, 17))
        initial_weights[0, [10, 11, 14, 15]] = kickto_weights[0, 1:]
        initial_weights[1, [10, 11, 14, 15]] = kickto_weights[1, 1:]
        initial_weights[2, 16] = shoot_goal_left_weights[1]
        initial_weights[3, 16] = shoot_goal_right_weights[1]

        initial_bias = np.zeros((4,))
        initial_bias[0] = kickto_weights[0, 0]
        initial_bias[1] = kickto_weights[1, 0]
        initial_bias[2] = shoot_goal_left_weights[0]
        initial_bias[3] = shoot_goal_right_weights[0]
        env = GoalFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)
        env = ScaledStateWrapper(env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.spaces[0].shape[0]

    discrete_action_dim = env.action_space.spaces[0].n
    action_parameter_sizes = np.array(
        [env.action_space.spaces[i].shape[0] for i in range(1, discrete_action_dim + 1)])
    parameter_action_dim = int(action_parameter_sizes.sum())
    discrete_emb_dim = discrete_action_dim
    parameter_emb_dim = parameter_action_dim
    max_action = 1.0
    print("state_dim", state_dim)
    print("discrete_action_dim", discrete_action_dim)
    print("parameter_action_dim", parameter_action_dim)

    kwargs = {
        "state_dim": state_dim,
        "discrete_action_dim": discrete_action_dim,
        "parameter_action_dim": parameter_action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "min_std": args.min_std,
        "max_std": args.max_std,
        "uniform_exploration_steps": args.exploration_steps,
        "interpolation": bool(args.interpolation),
    }

    # Initialize policy
    policy = da_ac.DAAC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, discrete_action_dim=discrete_action_dim,
                                       parameter_action_dim=parameter_action_dim,
                                       all_parameter_action_dim=parameter_action_dim,
                                       discrete_emb_dim=discrete_emb_dim,
                                       parameter_emb_dim=parameter_emb_dim,
                                       max_size=int(1e5))

    state, done = env.reset(), False
    episode_reward = 0

    total_reward = 0.
    returns = []
    Reward = []
    Reward_100 = []
    max_steps = 150
    cur_step = 0
    Test_Reward_100 = []
    Test_epioside_step_100 = []
    Test_success_rate_100 = []
    total_timesteps = 0
    t = 0
    while total_timesteps < args.max_timesteps:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        discrete_action, all_parameter_action, probs, mean, log_std = policy.select_action(state)
        offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
        parameter_action = all_parameter_action[offset:offset + action_parameter_sizes[discrete_action]]

        action = pad_action(discrete_action, parameter_action)
        episode_reward = 0.
        for i in range(max_steps):
            cur_step = cur_step + 1
            total_timesteps += 1
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            # Store one-hot encoded discrete action for replay buffer
            discrete_action_onehot = np.eye(discrete_action_dim)[discrete_action]
            replay_buffer.add(state, discrete_action=probs, 
                              parameter_action=mean, 
                              all_parameter_action=log_std,
                              discrete_emb=discrete_action_onehot,
                              parameter_emb=all_parameter_action,
                              next_state=next_state,
                              state_next_state=None,
                              reward=reward, done=terminal)

            state = next_state
            discrete_action, all_parameter_action, probs, mean, log_std = policy.select_action(next_state)
            offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
            parameter_action = all_parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
            action = pad_action(discrete_action, parameter_action)

            if cur_step >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)
            episode_reward += reward

            if total_timesteps % args.eval_freq == 0:
                print(
                    '{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(total_timesteps), total_reward / (t + 1),
                                                           np.array(returns[-100:]).mean()))

                while not terminal:
                    state = np.array(state, dtype=np.float32, copy=False)
                    discrete_action, all_parameter_action, probs, mean, log_std = policy.select_action(state)
                    offset = np.array([action_parameter_sizes[i] for i in range(discrete_action)], dtype=int).sum()
                    parameter_action = all_parameter_action[offset:offset + action_parameter_sizes[discrete_action]]
                    action = pad_action(discrete_action, parameter_action)
                    (state, _), reward, terminal, _ = env.step(action)

                Reward.append(total_reward / (t + 1))
                Reward_100.append(np.array(returns[-100:]).mean())
                Test_Reward, Test_epioside_step, Test_success_rate = evaluate(env, policy, action_parameter_sizes,
                                                           episodes=100)
                Test_Reward_100.append(Test_Reward)
                Test_epioside_step_100.append(Test_epioside_step)
                Test_success_rate_100.append(Test_success_rate)
            if terminal:
                break

        t += 1
        returns.append(episode_reward)
        total_reward += episode_reward

    print("save txt")
    dir = "result/DAAC/goal"
    data = "log"
    redir = os.path.join(dir, data)
    if not os.path.exists(redir):
        os.makedirs(redir)
    print("redir", redir)
    title2 = "Reward_100_da_ac_goal_"
    title3 = "Test_Reward_100_da_ac_goal_"
    title4 = "Test_epioside_step_100_da_ac_goal_"
    title5 = "Test_success_rate_100_da_ac_goal_"
    np.savetxt(os.path.join(redir, title2 + "{}".format(str(args.seed) + ".csv")), Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title3 + "{}".format(str(args.seed) + ".csv")), Test_Reward_100, delimiter=',')
    np.savetxt(os.path.join(redir, title4 + "{}".format(str(args.seed) + ".csv")), Test_epioside_step_100,
               delimiter=',')
    np.savetxt(os.path.join(redir, title5 + "{}".format(str(args.seed) + ".csv")), Test_success_rate_100,
               delimiter=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DA-AC")  # Policy name (DA-AC)
    parser.add_argument("--env", default='Goal-v0')  # Goal environment
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--ending_seed", default=-1, type=int)  # Number of seeds to run
    parser.add_argument("--start_timesteps", default=128, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=500, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=300000, type=float)  # Max time steps to run environment for
    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--min_std", default=0.05, type=float)  # Minimum standard deviation for exploration
    parser.add_argument("--max_std", default=0.2, type=float)  # Maximum standard deviation
    parser.add_argument("--exploration_steps", default=5000, type=int)  # Number of steps with uniform random actions
    parser.add_argument("--interpolation", default=1, type=int)  # Whether to use interpolation for critic update
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name

    args = parser.parse_args()
    starting_seed = args.seed
    if args.ending_seed == -1:
        args.ending_seed = starting_seed + 1
    for i in range(starting_seed, args.ending_seed):
        args.seed = i
        run(args)

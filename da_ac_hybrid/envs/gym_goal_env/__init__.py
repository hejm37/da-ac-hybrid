from gym.envs.registration import register

register(
    id="Goal-v0",
    entry_point="envs.gym_goal_env.goal_env:GoalEnv",
)

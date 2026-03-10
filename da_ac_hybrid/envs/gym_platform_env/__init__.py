from gym.envs.registration import register

register(
    id="Platform-v0",
    entry_point="envs.gym_platform_env.platform_env:PlatformEnv",
)

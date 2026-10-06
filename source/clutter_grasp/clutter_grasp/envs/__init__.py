"""Isaac Lab environments of the clutter grasping protocol.

Importing this module (after Isaac Sim is launched) registers:

    ClutterGrasp-Contactile-Benchmark-v0   Contactile hand, per-object PPO or external controller
    ClutterGrasp-PandaIK-Benchmark-v0      Franka Panda + parallel gripper, scripted IK + GG-CNN grasp pose
    ClutterGrasp-DataCollection-v0         random cluttered scenes for the clutter classifier dataset
"""

import gymnasium as gym

gym.register(
    id="ClutterGrasp-Contactile-Benchmark-v0",
    entry_point=f"{__name__}.benchmark_env:BenchmarkEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.benchmark_env_cfg:BenchmarkEnvCfg"},
)

gym.register(
    id="ClutterGrasp-PandaIK-Benchmark-v0",
    entry_point=f"{__name__}.panda_ik_benchmark_env:PandaIKBenchmarkEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.panda_ik_benchmark_env_cfg:PandaIKBenchmarkEnvCfg"},
)

gym.register(
    id="ClutterGrasp-DataCollection-v0",
    entry_point=f"{__name__}.data_collection_env:DataCollectionEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.data_collection_env_cfg:DataCollectionEnvCfg"},
)

# Change logs

current version: 0.1.7

## 0.1.7

1. Change _get_penalty_base_on_steps_left()_ in termination_base.py, now it can calculated penalty with rl_gamma=1.0.
2. Add two interfaces to the **task** module, _get_reach_target_terminations()_ and _is_reach_target_termination(termination_function_obj)_.
3. Remove redundant imports from python scritps.
4. Add a new section, **Research Areas supported by FlyCraft**, to README.md.

## 0.1.6

Update terminations: RT_single_step, NOBR, ES, C, CMA. Now they are calculated by the **next observations**.

## 0.1.5

Add a dense termination, **DenseRewardBasedOnAngle**, which is based on the angle error of velocity vector only.

## 0.1.3

Add some default configurations.

## 0.1.1

Change the configuration passing method. Currently, there are four ways to pass configuration during environment initialization. This change does not affect compatibility, and the training code based on the old version can still run in this version.

```python
# 1.use default configurations
env = gym.make('FlyCraft-v0')

# 2.pass configurations through config_file (Path or str)
env = gym.make('FlyCraft-v0', config_file=PROJECT_ROOT_DIR / "configs" / "NMR.json")

# 3.pass configurations through custom_config (dict), this method will load default configurations from default path, then update the default config with custom_config
env = gym.make(
    'FlyCraft-v0', 
    custom_config={
        "task": {
            "control_mode": "end_to_end_mode",
        }
    }
)

# 4.pass configurations through both config_file and custom_config. FlyCraft load config from config_file firstly, then update the loaded config with custom_config
env = gym.make(
    'FlyCraft-v0',
    config_file=PROJECT_ROOT_DIR / "configs" / "NMR.json",
    custom_config={
        "task": {
            "control_mode": "end_to_end_mode",
        }
    }
)
```

## 0.1.0

add a configuration that can specify whether training a guidance law model or a end-to-end model.

```json
    env_config = {
        ...
        "task": {
            "control_mode": "guidance_law_mode",  # or "end_to_end_mode"
            ...
        }
        ...
    }
```

# Change logs

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

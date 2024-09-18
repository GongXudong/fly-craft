# Change logs

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

## 0.1.1


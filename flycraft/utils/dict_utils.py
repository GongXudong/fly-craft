def update_nested_dict(d1, d2):
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            update_nested_dict(d1[key], value)
        else:
            d1[key] = value


def print_dict_structure(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + f"{key}: {type(value)}")
        if isinstance(value, dict):
            print_dict_structure(value, indent + 4)
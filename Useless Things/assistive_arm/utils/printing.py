
def print_dict_structure(d: dict, indent=0):
    """ Print dict structure
    Args:
        d (dict): target dict
        indent (int, 0): Choose indentation. Defaults to 0.
    """
    for key, value in d.items():
        print(' ' * indent + f"{key}: {type(value)}")
        if isinstance(value, dict):
            print_dict_structure(value, indent + 4)
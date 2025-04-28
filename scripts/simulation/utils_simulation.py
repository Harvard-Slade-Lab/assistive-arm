import os, sys
import logging
import importlib

def load_params():
    if len(sys.argv) != 2:
        error_logger("Include Parameter file.")
    param_path = sys.argv[1]
    if not os.path.isfile(param_path):
        error_logger("Parameter file not found.")
    return importlib.import_module(param_path[:-3]).PARAMS

def error_logger(txt):
    logging.error(txt)
    sys.exit()
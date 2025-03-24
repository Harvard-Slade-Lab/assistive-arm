import timeit
import logging


def get_logger() -> logging.Logger:
    # TODO Add general form of a logger
    pass


def print_elapsed_time(func, logger: logging.Logger=None) -> None:
    """ Decorator to log the elapsed time of a function

    Args:
        func (_type_): function to be decorated
        logger (logging.Logger): ñpgger to be used
    """
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        if logger:
            logger.info(f"Elapsed time: {end_time - start_time}")
        print(f"Elapsed time for '{func.__name__}': {end_time - start_time}")
        return result

    return wrapper
    
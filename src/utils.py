#!/usr/bin/env -S python -u

import time
import functools


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"[{func.__name__}: {elapsed_time:0.4f} s]", flush=True)
        return value

    return wrapper_timer

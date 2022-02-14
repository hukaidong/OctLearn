import subprocess
import logging
from multiprocessing import Pool
from os import environ, makedirs
from subprocess import Popen

import numpy as np

logger = logging.getLogger(__name__)

def steersim_call(query, env):
    logging.debug(f"pool is called with env {env}")
    steersim_command_path = env["SteersimCommandPath"]
    steersim_command_exec = env["SteersimCommandExec"]
    p = Popen(steersim_command_exec.split(), cwd=steersim_command_path,
              stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=None, env=env)
    stdout_line, stderr_line = p.communicate(input=query.encode())
    logging.debug('From subprocess: %r', stdout_line)
    p.wait()


def steersim_call_parallel(queries, generate_for_testcases=False):
    """
    Steersim arguments must be a list of numbers [len_query, len_parameters]
    """
    SteersimRecordPath = "SteersimRecordPath"
    env = environ.copy()
    if generate_for_testcases:
        env[SteersimRecordPath] = env[SteersimRecordPath] + "/test"

    makedirs(env[SteersimRecordPath], exist_ok=True)
    queries = np.clip(queries, 0, 1)
    query_strings = [" ".join([str(x) for x in numbers]) for numbers in queries]

    with Pool() as p:
        p.starmap(steersim_call, [(q, env) for q in query_strings])

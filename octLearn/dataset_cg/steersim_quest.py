from os import environ, makedirs
from subprocess import Popen, PIPE, STDOUT, DEVNULL
from multiprocessing import Pool


def steersim_call(query, env):
    steersim_command_path = env["SteersimCommandPath"]
    steersim_command_exec = env["SteersimCommandExec"]
    p = Popen(steersim_command_exec.split(), cwd=steersim_command_path,
              stdout=DEVNULL, stdin=PIPE, stderr=STDOUT, env=env)
    p.communicate(input=query.encode())
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
    query_strings = [" ".join([str(x) for x in numbers]) for numbers in queries]
    with Pool() as p:
        p.starmap(steersim_call, [(q, env) for q in query_strings])

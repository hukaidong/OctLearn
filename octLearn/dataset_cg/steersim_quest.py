from os import environ
from subprocess import Popen, PIPE, STDOUT
from multiprocessing import Pool


def steersim_call(query):
    steersim_command_path = environ["SteersimCommandPath"]
    steersim_command_exec = environ["SteersimCommandExec"]
    p = Popen(steersim_command_exec.split(), cwd=steersim_command_path,
              stdout=None, stdin=PIPE, stderr=STDOUT)
    p.communicate(input=query.encode())
    p.wait()


def steersim_call_parallel(queries):
    """
    Steersim arguments must be a list of numbers [len_query, len_parameters]
    """
    query_strings = [" ".join([str(x) for x in numbers]) for numbers in queries]
    with Pool() as p:
        p.map(steersim_call, query_strings)

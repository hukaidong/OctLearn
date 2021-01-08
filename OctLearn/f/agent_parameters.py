import numpy


def NormalizeAgentParameters(document):
    agent_parameters = document['agent parameters']
    params = numpy.zeros([len(agent_parameters), len(agent_parameters[0]['agent parameters'])])
    parameter_means = numpy.array([1.2, 8.0, 90.0, 0.75])
    for d in agent_parameters:
        aid, par = d.values()
        params[aid, :] = (par / parameter_means) - 1
    return params


def DenormalizeAgentParameters(array):
    parameter_means = numpy.array([1.2, 8.0, 90.0, 0.75])
    return (array + 1) * parameter_means

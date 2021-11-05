import numpy

def GetAgentParamConfigurations():
    pmin = numpy.array([0.5, 4.0, 45.0, 0.3])
    pmax = numpy.array([2.0, 12.0, 135.0, 1.2])
    pmean = (pmin + pmax) / 2
    pspan = pmax - pmin
    return { 'min': pmin, 'max': pmax,  'avg': pmean, 'rng': pspan }

def NormalizeAgentParameters(document):
    agent_parameters = document['agent parameters']
    params = numpy.zeros([len(agent_parameters), len(agent_parameters[0]['agent parameters'])])
    
    c = GetAgentParamConfigurations()
    
    for d in agent_parameters:
        aid, par = d.values()
        params[aid, :] = (par - c['avg']) / c['rng']
    return params


def DenormalizeAgentParameters(array):
    c = GetAgentParamConfigurations()
    pred = array * c['rng'] + c['avg']
    return numpy.clip(pred, a_min=c['min'], a_max=c['max'])

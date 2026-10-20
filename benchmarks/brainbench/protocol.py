"""Creator whole-prompt mean NLL, explicitly conditioned on Nemotron chat template."""
import math
MODEL='nvidia/nvidia/nemotron-3.5-super-vl'
def mean_nll(response):
    assert response['model']==MODEL
    lp=response['prompt_logprobs']
    assert len(lp)==response['usage']['prompt_tokens'] and lp[0] is None
    assert len(lp)>1 and all(isinstance(x,dict) and len(x)==1 for x in lp[1:])
    vals=[float(next(iter(x.values()))['logprob']) for x in lp[1:]]
    assert all(math.isfinite(v) and v<=0 for v in vals)
    return -sum(vals)/len(vals)
def score_pair(values,target):
    assert len(values)==2 and all(math.isfinite(v) for v in values)
    predicted=-1 if values[0]==values[1] else int(values[1]<values[0])
    return {'reward':float(predicted==target),'predicted':predicted,'mean_nll':values,'tie':predicted==-1}

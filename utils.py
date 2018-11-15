import numpy as np
# the stepper generator
def stepper(max_lr , min_lr , n_iter , func):
  n=1
  while(n<=n_iter):
    yield func(max_lr,min_lr,n/n_iter)
    n+=1
    
# Annealing functions
def annealing_no(start, end, pct):
    "No annealing, always return `start`."
    return start
  
def annealing_linear(start, end, pct):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end-start)
  
def annealing_exp(start, end, pct):
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start * (end/start) ** pct

def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out

# smoothing function
def calculateSmmothingValue(beta):
  n ,mov_avg=0,0
  while True :
    n+=1
    value = yield
    mov_avg = beta*mov_avg +(1-beta)*value
    smooth = mov_avg / (1 - beta **n )
    yield smooth
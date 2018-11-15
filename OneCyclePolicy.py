from keras.callbacks import Callback
import keras.backend  as K
import keras

from utils import calculateSmmothingValue , annealing_linear , annealing_cos, stepper

class OneCycleScheduler(Callback):

  def __init__(self,num_iteration,num_epochs,max_lr, momentum = (0.95,0.85) , div_factor=25 , pct_start=0.3):
    self.max_lr = max_lr
    self.momentum = momentum
    self.div_factor =div_factor
    self.pct_start = pct_start
    self.num_iteration = num_iteration
    self.num_epochs = num_epochs
    
    """ This callback implements a cyclical learning rate policy (CLR).
        This is a special case of Cyclic Learning Rates, where we have only 1 cycle.
        After the completion of 1 cycle, the learning rate will decrease rapidly to
        100th its initial lowest value.
        # Arguments:
            num_iteration: total no of iteration i.e total number of sample is dataset / batch_size
            num_epochs: Integer. Number of training epochs
            max_lr: Float. Initial learning rate. This also sets the
                starting learning rate (which will be 10x smaller than
                this), and will increase to this value during the first cycle.
            pct_start: Float. The percentage of all the epochs of training
                that will be dedicated to sharply decreasing the learning
                rate after the completion of 1 cycle. Must be between 0 and 1.
            momentum: Optional. Sets the maximum momentum (initial)
                value, which gradually drops to its lowest value in half-cycle,
                then gradually increases again to stay constant at this max value.
                Can only be used with SGD Optimizer.
                
        # Reference
            - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
            - [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
        """

  def step(self,*steps):
    return [stepper(*steps,n_iter,func) for (steps , (n_iter , func)) in zip(steps , self.phases)]

  def on_train_begin(self,logs=None):
    n = self.num_iteration*self.num_epochs
    a1 = int(n*self.pct_start)
    a2 = n-a1
    self.phases = ((a1 , annealing_linear) , (a2 , annealing_cos))
    self.min_lr = self.max_lr/self.div_factor
    self.lr_schedule = self.step((self.min_lr,self.max_lr) , (self.max_lr,self.min_lr/1e4))
    self.mom_schedule = self.step(self.momentum , self.momentum[::-1])
    
    self.lr = self.min_lr
    K.set_value(self.model.optimizer.lr,self.lr)
    self.mom = self.momentum[0]
    self.id=0
  
  def on_batch_end(self, batch, logs={}):
    try: 
      self.lr = next(self.lr_schedule[self.id])
      self.mom = next(self.mom_schedule[self.id])
    except:
      # when the current schedule is complete we move onto the next
      # schedule. (in 1-cycle there are two schedules)
      self.id=1

    if isinstance(self.model.optimizer , ( keras.optimizers.Adamax,keras.optimizers.Adam,keras.optimizers.Nadam)):
      K.set_value(self.model.optimizer.beta_1 , self.mom)
    elif isinstance(self.model.optimizer, keras.optimizers.SGD):
      K.set_value(self.model.optimizer.momentum,self.mom)
    K.set_value(self.model.optimizer.lr,self.lr)
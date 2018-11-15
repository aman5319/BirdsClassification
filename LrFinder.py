from pathlib import Path
import keras
from keras.models import load_model
from keras.callbacks import Callback
import keras.backend  as K
import matplotlib.pyplot as plt

from utils import calculateSmmothingValue , annealing_exp , stepper

class LRFinder(Callback):

    def __init__(self,max_iteration=100 , max_lr =10 ,base_lr=1e-7,loss_smoothing_beta=0.98):
        self.max_iteration = max_iteration
        self.max_lr = max_lr
        self.base_lr = base_lr
        self.losses=[]
        self.lrs=[]
        self.lr=0
        self.tmp= Path("tmp")
        self.loss_smoothing_beta = loss_smoothing_beta
        """
        This class uses the Cyclic Learning Rate history to find a
        set of learning rates that can be good initializations for the
        One-Cycle training proposed by Leslie Smith in the paper referenced
        below.
        A port of the Fast.ai implementation for Keras.
        # Note
        This requires that the model be trained for exactly 1 epoch.
        # Interpretation
        Upon visualizing the loss plot, check where the loss starts to increase
        rapidly. Choose a learning rate at somewhat prior to the corresponding
        position in the plot for faster convergence. This will be the maximum_lr lr.
        Choose the max value as this value when passing the `max_val` argument
        to OneCycleLR callback.
        Since the plot is in log-scale, you need to compute 10 ^ (-k) of the x-axis
        # Arguments:
            max_iteration: total no of iteration i.e total number of sample is dataset / batch_size
            base_lr: Float. Initial learning rate (and the minimum).
            max_lr: Float. Final learning rate (and the maximum).
            lr_scale: Can be one of ['annealing_exp', 'annealing_linear']. Chooses the type of
                scaling for each update to the learning rate during subsequent
                batches. Choose 'exp' for large range and 'linear' for small range.
            loss_smoothing_beta: Float. The smoothing factor for the moving
                average of the loss function.
        # References:
            - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
        """

    def on_train_begin(self, logs={}):
      self.step = stepper(self.base_lr , self.max_lr ,self.max_iteration , annealing_exp)       #anneal function
      self.smoothner = calculateSmmothingValue(self.loss_smoothing_beta) #smoothing function
      self.model.save_weights(self.tmp.name) # save the current state of model to reset after the training is done
     
    def on_batch_begin(self,batch,logs={}):
      self.lr = self.base_lr if batch == 0  else next(self.step)
      # smooth it and update the lr
      next(self.smoothner)
      self.lr = self.smoothner.send(self.lr)
      K.set_value(self.model.optimizer.lr , self.lr)
  
    def on_batch_end(self,batch,logs={}):
        # stop the training if lr is greater then max lr
        if self.lr >= self.max_lr:
          self.model.stop_training= True
        self.losses.append(logs.get("loss"))
        self.lrs.append(self.lr)
        
    def on_epoch_end(self, epoch, logs=None):
        self.model.stop_training= True
        
    def on_train_end(self,logs=None):
        self.model.load_weights(self.tmp.name)        # reset the model
        self.tmp.unlink() # delete the file
        plt.semilogx(self.lrs, self.losses) #plot it
        plt.show()


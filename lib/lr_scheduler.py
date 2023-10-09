import numpy as np
import tensorflow as tf
from keras.utils import io_utils
from keras import backend
from typing import Tuple

class Scheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler.
    """
    def __init__(self, batch_update=False, steps_per_epoch=None, verbose:bool=True, name="LearningRateScheduler"):
        super().__init__()
        self.verbose = verbose
        self.name = name
        self.update_per_batch = batch_update
        if(batch_update):
            assert type(steps_per_epoch) == int and int(steps_per_epoch) > 1, "invalid steps per epoch"
        self.steps_per_epoch = steps_per_epoch if batch_update else 1
        self.history = []

    def get_history(self):
        return self.history

    def on_train_batch_begin(self, batch, logs=None):
        if not self.update_per_batch: return
        lr = float(backend.get_value(self.model.optimizer.lr))
        lr = self.compute_lr(self.current_epoch, batch, lr, logs)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function ' + f"should be float. Got: {lr}")
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}")
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))

    def on_train_batch_end(self, batch, logs=None):
        self.history[-1].append(backend.get_value(self.model.optimizer.lr))

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.history.append([])
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(backend.get_value(self.model.optimizer.lr))
        if not self.update_per_batch:
            # do the update on the start of the epoch
            lr = self.compute_lr(epoch, 0, lr, logs)
            if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function ' + f"should be float. Got: {lr}")
            if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
                raise ValueError(f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}")
            backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
            if self.verbose:
                io_utils.print_msg(f"\nEpoch {epoch + 1}: {self.name} setting learning rate to {lr}.")
        else:
            if self.verbose:
                io_utils.print_msg(f"\nEpoch {epoch + 1}: {self.name} current learningrate = {lr}.")
                
    def on_epoch_end(self, epoch:int, logs:dict=None):
        logs = logs or {}
        logs["lr"] = backend.get_value(self.model.optimizer.lr)

    def compute_lr(self, epoch:int, batch:int, lr:float, logs:dict):
        raise NotImplementedError("Schedulers should override compute_lr")

class ChainedScheduler(Scheduler):
    def __init__(self, chain:dict=None, **kwargs):
        super().__init__(name="ChainedScheduler", **kwargs)
        self.schedulers:list[Scheduler] = []
        self.idx = []
        self.start_epochs = [0]
        self.last_has_been_added = False
        self.last_epoch = -1
        if(chain is not None):
            args = dict(batch_update=self.update_per_batch, steps_per_epoch =self.steps_per_epoch)
            for idx, element in enumerate(chain):
                name,params,duration = element["type"], element["params"], element["duration"]
                if(name == "warmup"):
                    self.append(GradualWarmupScheduler(**args, **params, verbose=self.verbose), duration)
                elif(name == "cooldown"):
                    self.append(GradualCooldownScheduler(**args, **params, verbose=self.verbose), duration)
                elif(name == "cosine"):
                    self.append(CosineAnnealingWarmRestarts(**args, **params, verbose=self.verbose), duration)
                elif(name=="expdecay"):
                    self.append(ExponentialDecay(**args, **params, verbose=self.verbose),duration)
                elif(name=="reduce"):
                    self.append(ReduceLROnPlateau(**args, **params, verbose=self.verbose), duration)
                elif(name=="constant"):
                    self.append(Constant(**args, **params, verbose=self.verbose), duration)
                elif(name=="step"):
                    self.append(Step(**args, **params, verbose=self.verbose), duration)
                else:
                    raise ValueError(f"Unknown LR schedule `{name}`.")
    
    def set_model(self, model):
        super().set_model(model)
        for scheduler in self.schedulers:
            scheduler.set_model(model)
    def set_params(self, params):
        super().set_params(params)
        for scheduler in self.schedulers:
            scheduler.set_params(params)
 
    def on_epoch_begin(self, epoch, logs=None):
        if(epoch < len(self.idx)):
            idx = self.idx[epoch]
        else:
            if not self.last_has_been_added:
                print("ChainedScheduler ran out of schedules! Using the last one...")
            idx = len(self.schedulers)-1
        self.name = "ChainedScheduler/"+self.schedulers[idx].name
        self.current_scheduler:Scheduler = self.schedulers[idx]
        self.current_epoch_offset = self.start_epochs[idx]#+(1 if self.update_per_batch else 0)
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        scheduler, start = self.get_at_epoch(epoch)
        scheduler.on_epoch_end(epoch-start, logs)

    def compute_lr(self, epoch:int , batch:int, lr:float, logs:dict=None):
        if(self.last_epoch!=epoch):
            self.current_scheduler, self.current_epoch_offset = self.get_at_epoch(epoch)
            self.name = self.current_scheduler.name
            self.last_epoch = epoch
        return self.current_scheduler.compute_lr(epoch-self.current_epoch_offset, batch, lr, logs=logs)

    def append(self,scheduler:Scheduler, duration:int|None):
        if self.last_has_been_added:
            raise ValueError("The previous scheduler has an infinite duration, cannot add another")
        self.schedulers.append(scheduler)
        if(duration is None):
            self.last_has_been_added = True
            self.start_epochs.append(None)
        else:
            self.idx.extend([len(self.schedulers)-1]*duration) #so self.idx[epoch] returns the index of the scheduler of that epoch
            self.start_epochs.append(self.start_epochs[-1]+duration) #start epoch of the next scheduler

    def get(self,idx):
        if(idx>len(self.schedulers)):
            raise IndexError("List index out of range")
        return (self.schedulers[idx], self.start_epochs[idx])
    
    def get_at_epoch(self,epoch) -> tuple[Scheduler, int]:
        if(epoch < len(self.idx)):
            #return (self.schedulers[self.idx[epoch]], self.start_epochs[self.idx[epoch]]+(1 if self.update_per_batch else 0))
            return (self.schedulers[self.idx[epoch]], self.start_epochs[self.idx[epoch]])
        if(self.last_has_been_added):
            return (self.schedulers[-1], self.start_epochs[-2]+(1 if self.update_per_batch else 0))
        raise IndexError("No scheduler for that epoch")

class GradualCooldownScheduler(Scheduler):
    def __init__(self, start_lr:float|None=None, multiplier:float=1.0,cooldown_epochs:int=10, final_lr:float|None=None, **kwargs):
        super().__init__(**kwargs, name="GradualCooldownScheduler")
        assert multiplier<=1., "Multiplier should be <=1"
        if(multiplier==1 and final_lr is None):
            raise ValueError("If multiplier==1, final_lr needs to be provided.")
        if(final_lr is not None and start_lr is not None):
            assert final_lr < start_lr, "final_lr needs to be smaller than start_lr"
        self.cooldown_epochs = cooldown_epochs
        self.multiplier = multiplier
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.finished = False
        self.cooldown_steps = cooldown_epochs*self.steps_per_epoch
        print(f"{self.cooldown_steps=}")
        self.showed_error = False

    def compute_lr(self, epoch:int, batch:int, lr:float, logs:dict=None):
        if(self.start_lr is None):
            self.start_lr = lr
        # We always calculate the new learning rate based on the epoch, not based on the previous lr
        if(epoch > self.cooldown_epochs):
            if(self.verbose and not self.showed_error):
                self.showed_error = True # make sure we only print this once per epoch
                print(f"GradualCooldownScheduler: epoch {epoch} - ran out of cooldown epochs, return same lr")
            return lr
        step = epoch*self.steps_per_epoch+batch
        if self.multiplier == 1.0:
            lr = self.start_lr+(self.final_lr-self.start_lr)*step/self.cooldown_steps
        else:
            lr = self.start_lr*((self.multiplier-1)*step/self.cooldown_steps+1)
        return lr

    def on_epoch_end(self, epoch: int, logs: dict = None):
        self.showed_error = False
        return super().on_epoch_end(epoch, logs)

class GradualWarmupScheduler(Scheduler):
    def __init__(self, start_lr:float|None=None, multiplier:float=1.0,warmup_epochs:int=10, final_lr:float|None=None, **kwargs):
        super().__init__(**kwargs, name="GradualWarmupScheduler")
        assert multiplier>=1., "Multiplier should be >=1"
        if(multiplier==1 and final_lr is None):
            raise ValueError("If multiplier==1, final_lr needs to be provided.")
        if(final_lr is not None and start_lr is not None):
            assert final_lr > start_lr, "final_lr needs to be larger than start_lr"
        self.warmup_epochs = warmup_epochs
        self.multiplier = multiplier
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.finished = False
        self.warmup_steps = warmup_epochs*self.steps_per_epoch
        print(f"{self.warmup_steps=}")
        self.showed_error = False

    def compute_lr(self, epoch:int, batch:int, lr:float, logs:dict):
        if(self.start_lr is None):
            self.start_lr = lr
        # We always calculate the new learning rate based on the epoch, not based on the previous lr
        if(epoch > self.warmup_epochs):
            if(self.verbose and not self.showed_error):
                self.showed_error = True # make sure we show this only once per epoch
                print(f"GradualWarmupScheduler: epoch {epoch} - ran out of warmup epochs, return same lr")
            return lr
        step = epoch*self.steps_per_epoch+batch
        if self.multiplier == 1.0:
            lr = self.start_lr+(self.final_lr-self.start_lr)*step/self.warmup_steps
        else:
            lr = self.start_lr*((self.multiplier-1)*step/self.warmup_steps+1)
        return lr
    
    def on_epoch_end(self, epoch: int, logs: dict = None):
        self.showed_error = False
        return super().on_epoch_end(epoch, logs)

class CosineAnnealingWarmRestarts(Scheduler):
    def __init__(self, start_lr:float|None=None, T_0:int=10, T_mult:int=1, eta_min:float=0., gamma:float=1, eta_gamma:float=1, last_epoch:int=-1,
                 **kwargs):
        super().__init__(**kwargs, name="CosineAnnealingWarmRestarts")
        assert T_0 > 0 and type(T_0) is int, "T_0 should be a integer>1, got " + str(T_0)
        assert T_mult >= 1 and type(T_mult) is int, "T_mult should be a integer>=1, got " + str(T_mult)
        assert 0 < gamma <= 1, "gamma should be >0 and <=1. Got " + str(gamma)
        assert 0 < eta_gamma <= 1, "eta_gamma should be >0 and <=1. Got " + str(eta_gamma)
        self.T_0 = T_0
        self.T_i = T_0
        self.restarts = 0
        self.T_mult = T_mult
        self.T_curr = -1
        self.gamma = gamma
        self.eta_gamma = eta_gamma

        self.T_curr_steps = -1
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.last_lr = self.start_lr = start_lr
        
    def next_epoch(self):
        self.T_curr += 1
        self.last_epoch += 1
        if(self.T_curr > self.T_i):
            self.restarts += 1
            self.start_lr *= self.gamma
            self.eta_min *= self.eta_gamma
            self.T_i = self.T_0*(self.T_mult**self.restarts)-1 #-1 to deal with the > sign (instead of >=)
            self.T_curr = 0
        self.T_curr_steps = self.T_curr * self.steps_per_epoch
        self.total_steps = (self.T_i+(1 if self.update_per_batch else 0))*self.steps_per_epoch

    def compute_lr(self,epoch:int, batch:int, lr:float, logs:dict):
        if(self.start_lr is None):
            self.start_lr = lr
            self.last_lr = lr
        #We calculate the lr based on the previous lr, so we need to make sure to 'skip ahead' if the 
        # epoch is not exactly last_epoch
        if(epoch != self.last_epoch):
            while(epoch > self.last_epoch): self.next_epoch()
        self.last_lr = self.eta_min+(self.start_lr-self.eta_min)*(1+np.cos(np.pi*(self.T_curr_steps+batch)/(self.total_steps)))/2.
        return self.last_lr

class ExponentialDecay(Scheduler):
    def __init__(self, start_lr:float=None, gamma:float=0.9,last_epoch:int=-1,**kwargs):
        super().__init__(**kwargs,name="ExponentialDecay")
        self.start_lr = start_lr
        self.gamma_step = gamma**(1/self.steps_per_epoch)
        self.last_epoch = last_epoch
        self.last_batch = 0

    def compute_lr(self, epoch:int, batch:int, lr:float, logs:dict):
        if(self.start_lr is None):
            self.start_lr = lr
        #try to just update the previous lr
        if(epoch==self.last_epoch and batch == self.last_batch+1): lr = lr*self.gamma_step
        elif(epoch==self.last_epoch+1 and batch == 0): lr = lr*self.gamma_step
        else: lr = self.start_lr*(self.gamma_step**((epoch-self.last_epoch)*self.steps_per_epoch+batch-self.last_batch))
        self.last_epoch = epoch
        self.last_batch = batch
        return lr

class ReduceLROnPlateau(Scheduler):
    def __init__(self, monitor='val_loss', start_lr:float=None, factor:float=0.1, min_lr:float=0, min_delta=3e-4, patience=10, mode='min', **kwargs):
        super().__init__(**kwargs, name="ReduceLROnPlateau")
        self.start_lr = start_lr
        self.factor = factor
        self.monitor = monitor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience*self.steps_per_epoch
        self.comp = (lambda a,b: a-b>min_delta) if mode=='max' else (lambda a,b: b-a>min_delta)
        self.best_val = -np.inf if mode=='max' else np.inf
        self.wait = 0
        self.logs_prev_epoch = {}

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.logs_prev_epoch = logs

    def compute_lr(self, epoch:int, batch:int, lr:float, logs:dict):
        if(self.start_lr is None):
            self.start_lr = lr
        val = self.logs_prev_epoch.get(self.monitor, None)
        if(val is not None):
            if(self.comp(val,self.best_val)):
                #io_utils.print_msg(f"`{self.monitor}` improved from {self.best_val} to {val}")
                self.best_val = val
                self.wait = 0
            else:
                self.wait += 1
                if(self.wait >= self.patience):
                    new_lr = max(self.factor*lr,self.min_lr)
                    io_utils.print_msg(f"`{self.monitor}` did not improve from {self.best_val} for {self.wait} steps, reducing LR")
                    self.wait = 0
                    return new_lr
                else:
                    pass#io_utils.print_msg(f"`{self.monitor}` did not improve from {self.best_val} for {self.wait} epochs")
            return lr
        else:
            return lr

class Constant(Scheduler):
    def __init__(self, start_lr:float=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_lr = start_lr

    def compute_lr(self, epoch:int, batch:int, lr:float, logs:dict):
        self.start_lr = self.start_lr or lr
        return self.start_lr

class Step(Scheduler):
    def __init__(self, start_lr:float=None, factor:float=0.1, T_wait:int=4, T_duration:int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_lr = start_lr
        self.T_wait = T_wait
        self.duration = T_duration*self.steps_per_epoch
        self.factor = factor
        self.T = 0
        self.last_epoch, self.last_batch = -1, -1
        self.current_lr = self.start_lr
        self.reducing = None

    def compute_lr(self, epoch:int, batch:int, lr:float, logs:dict):
        if self.start_lr is None:
            self.start_lr = lr
            self.current_lr = lr
        if(epoch != self.last_epoch):
            self.reducing = None
            self.T += epoch - self.last_epoch
            self.last_epoch = epoch
            self.last_batch = 0
            if(self.T > self.T_wait):
                self.T = 0
                if(self.update_per_batch): self.reducing = (self.factor-1)*self.current_lr/self.duration
                else: self.current_lr *= self.factor
        if(self.reducing):
            self.current_lr += self.reducing*(batch-self.last_batch)
            self.last_batch = batch
        return self.current_lr
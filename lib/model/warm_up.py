from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR

class lr_convert():
    def __init__(self, config, optimizer):
        self.lr = config.lr
        self.start_epoch = config.start_epoch
        self.num_epochs = config.num_epochs
        self.optimizer = optimizer
        self.milestones = [10, 20, 40]
        self._sched_map = {
            '_warm_up'    : self._warm_up,
            '_plateau'    : self._plateau,     
            '_cosine'     : self._cosine,
            '_plateau_low': self._plateau_low,
            }
    def _warm_up(self):
        warmup = LinearLR(self.optimizer,
                        start_factor=1e-6/self.lr,
                        end_factor=1.0,
                        total_iters=10 - self.start_epoch)
        return warmup
    def _plateau(self):
        plateau = ConstantLR(
                        self.optimizer,
                        factor=1.0,           
                        total_iters=20 - max(self.start_epoch, 10)        
            )
        return plateau
    def _cosine(self):
        cosine = CosineAnnealingLR(
                        self.optimizer,
                        T_max=40 - max(self.start_epoch, 20),
                        eta_min=1e-6)   
        return cosine
    def _plateau_low(self):
        plateau_low = ConstantLR(
                        self.optimizer,
                        factor=1e-6 / self.lr,  
                        total_iters=self.num_epochs - max(self.start_epoch, 40)       
        )
        return plateau_low
    def pop_up(self):
        schedulers_name = ['_warm_up', '_plateau', '_cosine', '_plateau_low']
        added_milestones = sorted(self.milestones + [self.start_epoch])
        idx = added_milestones.index(self.start_epoch)
        self.milestones[idx:]
        return schedulers_name[idx:]

    def get_schedulers(self):
        schedulers_name = self.pop_up()
        schedulers = [self._sched_map[name]() for name in schedulers_name]
        return SequentialLR(self.optimizer,
                    schedulers=schedulers,
                    milestones=self.milestones)


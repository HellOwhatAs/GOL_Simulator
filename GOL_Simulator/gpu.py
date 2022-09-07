from numba import cuda
import numpy as np

@cuda.jit
def update_new(world,new_world):
    _i,_j=cuda.grid(2)
    ti,tj=cuda.gridsize(2)
    for i in range(_i+1,world.shape[0]-1,ti):
        for j in range(_j+1,world.shape[1]-1,tj):
            tmp=(world[i-1,j-1])+\
            (world[i-1,j])+\
            (world[i-1,j+1])+\
            (world[i,j-1])+\
            (world[i,j+1])+\
            (world[i+1,j-1])+\
            (world[i+1,j])+\
            (world[i+1,j+1])

            if tmp==3:new_world[i,j]=1
            elif tmp<2 or tmp>3:new_world[i,j]=0

class Simulator:
    def __init__(self,init_state:np.ndarray):
        expanded_state=np.zeros(tuple(i+2 for i in init_state.shape),dtype=np.bool8)
        expanded_state[1:-1,1:-1]=init_state
        self.field=cuda.to_device(expanded_state)
        self.new_field=cuda.device_array(self.field.shape,self.field.dtype)
    def run(self,rounds:int=1,tqdm=None):
        for _ in range(rounds) if tqdm is None else tqdm(range(rounds)):
            update_new[(min(256,self.field.shape[0]//32+1),min(256,self.field.shape[1]//32+1)),(32,32)](self.field,self.new_field)
            self.field.copy_to_device(self.new_field)
            cuda.synchronize()
        return self
    def __getitem__(self,x):
        ret=self.field[1:-1,1:-1][x]
        if hasattr(ret,"copy_to_host"):return ret.copy_to_host()
        return ret
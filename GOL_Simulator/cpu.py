from numba import njit
import numpy as np

@njit
def update_new(world,new_world):
    for i in range(1,world.shape[0]-1):
        for j in range(1,world.shape[1]-1):
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
        self.field=np.zeros(tuple(i+2 for i in init_state.shape),dtype=np.bool8)
        self.field[1:-1,1:-1]=init_state
        self.new_field=np.empty_like(self.field)
    def run(self,rounds:int=1,tqdm=None):
        for _ in range(rounds) if tqdm is None else tqdm(range(rounds)):
            update_new(self.field,self.new_field)
            self.field[:]=self.new_field
        return self
    @property
    def result(self):
        return self.field
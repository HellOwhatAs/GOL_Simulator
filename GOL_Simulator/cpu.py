from numba import njit
import numpy as np
from typing import Union

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
        self.__field:np.ndarray=np.zeros(tuple(i+2 for i in init_state.shape),dtype=np.bool8)
        self.__field[1:-1,1:-1]=init_state
        self.__new_field=(self.__field).copy()
    def run(self,rounds:int=1,tqdm=None):
        for _ in range(rounds) if tqdm is None else tqdm(range(rounds)):
            update_new(self.__field,self.__new_field)
            self.__field[:]=self.__new_field
        return self
    def __getitem__(self,x)->Union[np.ndarray,np.bool_]:
        return self.__field[1:-1,1:-1][x]
    def __setitem__(self,__s,__o):
        self.__field[1:-1,1:-1][__s]=__o
        self.__new_field[1:-1,1:-1][__s]=__o
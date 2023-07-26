from numba import njit, prange
import numpy as np
import numpy.typing
from typing import Union

@njit(parallel=True)
def update_new(world, new_world):
    for i in prange(world.shape[0]):
        for j in range(world.shape[1]):
            tmp = (
                (world[(i-1) % world.shape[0], (j-1) % world.shape[1]])+
                (world[(i-1) % world.shape[0], (j  ) % world.shape[1]])+
                (world[(i-1) % world.shape[0], (j+1) % world.shape[1]])+
                (world[(i  ) % world.shape[0], (j-1) % world.shape[1]])+
                (world[(i  ) % world.shape[0], (j+1) % world.shape[1]])+
                (world[(i+1) % world.shape[0], (j-1) % world.shape[1]])+
                (world[(i+1) % world.shape[0], (j  ) % world.shape[1]])+
                (world[(i+1) % world.shape[0], (j+1) % world.shape[1]])
            )

            if tmp == 3: new_world[i,j]=1
            elif tmp<2 or tmp>3: new_world[i,j]=0

class Simulator:
    def __init__(self, init_state: numpy.typing.NDArray):
        self.__field = np.zeros(init_state.shape, dtype=np.bool8)
        self.__field[:] = init_state
        self.__new_field = (self.__field).copy()
    
    def run(self, rounds: int=1, tqdm = None):
        for _ in range(rounds) if tqdm is None else tqdm(range(rounds)):
            update_new(self.__field, self.__new_field)
            self.__field[:] = self.__new_field
        return self
    
    def __getitem__(self, x) -> Union[numpy.typing.NDArray[np.uint8], np.bool8]:
        return self.__field[x]
    
    def __setitem__(self, __s, __o):
        self.__field[__s] = __o
        self.__new_field[__s] = __o
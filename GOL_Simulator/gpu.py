from numba import cuda
import numpy as np
import numpy.typing
from typing import Union
import math

@cuda.jit
def update_new(world, new_world):
    _i, _j = cuda.grid(2)
    ti, tj = cuda.gridsize(2)
    for i in range(_i, world.shape[0], ti):
        for j in range(_j, world.shape[1], tj):
            tmp= (
                (world[(i-1) % world.shape[0], (j-1) % world.shape[1]])+
                (world[(i-1) % world.shape[0], (j  ) % world.shape[1]])+
                (world[(i-1) % world.shape[0], (j+1) % world.shape[1]])+
                (world[(i  ) % world.shape[0], (j-1) % world.shape[1]])+
                (world[(i  ) % world.shape[0], (j+1) % world.shape[1]])+
                (world[(i+1) % world.shape[0], (j-1) % world.shape[1]])+
                (world[(i+1) % world.shape[0], (j  ) % world.shape[1]])+
                (world[(i+1) % world.shape[0], (j+1) % world.shape[1]])
            )

            if tmp == 3: new_world[i,j] = 1
            elif tmp<2 or tmp>3: new_world[i,j] = 0

class Simulator:
    def __init__(self, init_state: numpy.typing.NDArray):
        self.__field = cuda.to_device(init_state.astype(np.uint8))
        self.__new_field = cuda.to_device(init_state.astype(np.uint8))

        self.threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(init_state.shape[0] / self.threadsperblock[0])
        blockspergrid_y = math.ceil(init_state.shape[1] / self.threadsperblock[1])
        self.blockspergrid = (blockspergrid_x, blockspergrid_y)

    def run(self, rounds: int = 1, tqdm = None):
        for _ in range(rounds) if tqdm is None else tqdm(range(rounds)):
            update_new[self.blockspergrid, self.threadsperblock](self.__field, self.__new_field)
            cuda.synchronize()
            self.__field.copy_to_device(self.__new_field)
        return self

    def __getitem__(self,x) -> Union[numpy.typing.NDArray[np.uint8], np.bool8]:
        ret = self.__field[x]
        if hasattr(ret, "copy_to_host"): return ret.copy_to_host()
        return ret

    def __setitem__(self, __s, __o):
        self.__field[__s] = __o
        self.__new_field[__s] = __o
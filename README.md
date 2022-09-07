# GOL_Simulator
Simulator of GOL(Game of Life) with Python on both cpu and gpu(cuda)

# Usage
If [`numba.cuda`]([https://github.com/numba/numba](https://numba.readthedocs.io/en/stable/cuda/index.html) is avaliable for you, use `from GOL_Simulator.gpu import Simulator` (or cpu if you like it).  
Otherwise you can use `from GOL_Simulator.cpu import Simulator` only.  
(all API are the same between `gpu` and `cpu`)

# Examples
run [cpu_example.py](https://github.com/HellOwhatAs/GOL_Simulator/blob/main/cpu_example.py) to get [cpu_result.png](https://github.com/HellOwhatAs/GOL_Simulator/blob/main/cpu_result.png) which is a randomly (with all cells 95% probability to be dead and 5% to be alive) initialized `1000x1000` state after `200` iterations.

run [gpu_example.py](https://github.com/HellOwhatAs/GOL_Simulator/blob/main/cpu_example.py) to get [gpu_result.png](https://github.com/HellOwhatAs/GOL_Simulator/blob/main/cpu_result.png) which is the state in [cpu_result.png](https://github.com/HellOwhatAs/GOL_Simulator/blob/main/gpu_result.png) (actually it reads the state from this picture) after another `2000` iterations.


import cv2,numpy as np
from GOL_Simulator.gpu import Simulator
from tqdm import tqdm

s=Simulator(
    cv2.imread("cpu_result.png",0)
).run(2000,tqdm)

cv2.imwrite("gpu_result.png",s.result.astype(np.uint8)*255)
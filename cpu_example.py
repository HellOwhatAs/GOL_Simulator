import cv2,numpy as np
from GOL_Simulator.cpu import Simulator
from tqdm import tqdm

sim=Simulator(
    (np.random.random((10000,10000))>0.95)
).run(200,tqdm)
cv2.imwrite("cpu_result.png",sim[:].astype(np.uint8)*255)

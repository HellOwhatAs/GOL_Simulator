import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from GOL_Simulator.gpu import Simulator
from GOL_Simulator import rle

if __name__ == '__main__':

    win = pg.GraphicsLayoutWidget()
    win.show()
    win.setWindowTitle('Game of Life')
    view = win.addViewBox()
    view.setAspectLocked(True)

    img = pg.ImageItem(border='w')
    view.addItem(img)

    with open('./patterns/pattern (3).rle', 'r') as f:
        pattern = rle.loads(f.read())
    paddings = {
        '<': 10000,
        '>': 10000,
        'v': 100,
        '^': 100
    }
    field = np.zeros((pattern.shape[0] + (paddings['<'] + paddings['>']), pattern.shape[1] + (paddings['^'] + paddings['v'])), np.uint8)
    field[paddings['<']:-paddings['>'] if paddings['>'] else None, paddings['^']:-paddings['v'] if paddings['v'] else None] = pattern

    sim = Simulator(field)

    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    def updateData():
        global sim, img
        img.setImage(sim.run(11)[:])
        timer.start()
    timer.timeout.connect(updateData)
    updateData()

    pg.exec()

    with open('finish.rle', 'w') as f:
        f.write(rle.dumps(sim[:]))
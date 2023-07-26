import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from GOL_Simulator.gpu import Simulator
from GOL_Simulator import rle
from typing import Optional

def main(ifname: str, ofname: Optional[str] = None, rounds: int = 1,
         left_padding: int = 100, right_padding: int = 100, top_padding: int = 100, bottom_padding: int = 100):
    win = pg.GraphicsLayoutWidget()
    win.show()
    win.setWindowTitle('Game of Life')
    view = win.addViewBox()
    view.setAspectLocked(True)

    img = pg.ImageItem(border='w')
    view.addItem(img)

    with open(ifname, 'r') as f:
        pattern = rle.loads(f.read())
    
    paddings = {
        '<': left_padding,
        '>': right_padding,
        'v': bottom_padding,
        '^': top_padding
    }
    field = np.zeros((pattern.shape[0] + (paddings['<'] + paddings['>']), pattern.shape[1] + (paddings['^'] + paddings['v'])), np.uint8)
    field[paddings['<']:-paddings['>'] if paddings['>'] else None, paddings['v']:-paddings['^'] if paddings['^'] else None] = pattern

    sim = Simulator(field)

    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    def updateData():
        nonlocal sim, img
        img.setImage(sim.run(rounds)[:])
        timer.start()
    timer.timeout.connect(updateData)
    updateData()

    pg.exec()

    if ofname is not None:
        with open(ofname, 'w') as f:
            f.write(rle.dumps(sim[:]))

if __name__ == '__main__':
    import fire
    import sys
    fire.Fire(main, [
        './patterns/pattern.rle',
        '--rounds', '10',
        '--right_padding', '2000',
        '--top_padding', '1000'
    ] if sys.gettrace() else None)
import re
import numpy as np
from numpy.typing import NDArray
from itertools import groupby

def loads(s: str) -> NDArray[np.bool8]:
    lines = [new_line for line in s.split('\n') if (new_line := (line[:idx] if (idx := line.find('#')) != -1 else line).strip())]
    args = {arg[:idx].strip(): arg[idx+1:].strip() for arg in lines[0].split(',') if (idx := arg.find('=')) != -1}
    lines.pop(0)
    content = ''.join(lines)

    nums = re.findall(r'\d+', content)
    syms = re.split(r'\d+', content)

    _raw = [syms[0]]
    syms.pop(0)

    for num, sym in zip(nums, syms):
        _raw.append(int(num) * sym[0] + sym[1:])

    raw = ''.join(_raw)[:-1]
    raw_lines = raw.split('$')

    ret = np.zeros((int(args['y']), int(args['x'])), np.bool8)
    for idx, raw_line in enumerate(raw_lines):
        tmp = [0 if i == 'b' else 1 for i in raw_line]
        ret[idx][:len(tmp)] = tmp

    return np.rot90(ret, -1)

def dumps(arr: NDArray[np.bool8]) -> str:
    nz = np.nonzero(arr)
    arr = np.rot90(arr[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1])
    comment = '#C Generated by GOL_Simulator'
    args = {'x': arr.shape[1], 'y': arr.shape[0], 'rule': 'B3/S23'}
    lines = '$'.join(''.join('o' if i else 'b' for i in line) for line in arr)

    contents = []
    for char, iterator in groupby(lines):
        if char == '$' and contents and contents[-1][0] == 'b': contents.pop()
        contents.append((char, sum(1 for _ in iterator)))
    
    content = ''.join(f'{count}{char}' if count > 1 else char for char, count in contents)
    
    return '\n'.join((comment, ', '.join(f'{k} = {v}' for k, v in args.items()), content, '!'))


if __name__ == '__main__':
    with open('patterns/pattern.rle', 'r') as f:
        ret = loads(f.read())
    import cv2
    cv2.imwrite('output.png', ret.astype(np.uint8) * 255)
    assert (loads(dumps(ret)) == ret).all()
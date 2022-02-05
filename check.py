import os
import numpy as np
import multiprocessing as mp


def check(txt: str) -> int:
    error = False
    with open(txt, "r", encoding="utf-8") as f:
        record = f.readlines()
        for i, line in enumerate(record[6:], 7):
            if line.split()[1] == "和牌" and i != len(record):
                error = True
                break
    if error:
        os.remove(txt)
        return 1
    return 0

if __name__ == "__main__":
    human = "stdmahjong/raw"
    n, e = 0, 0
    print("start")
    for folded in os.listdir(human):
        folder = [os.path.join(human, folded, file) for file in os.listdir(os.path.join(human, folded))]
        n += len(folder)
        pool = mp.Pool(mp.cpu_count())
        res = list(pool.map(check, folder))
        pool.close()
        pool.join()
        e += np.sum(np.array(res, int))
        print("finished %d deleted %d" % (n, e))
    print("all %d deleted %d" % (n, e))

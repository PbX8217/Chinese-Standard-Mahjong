import os, json
import numpy as np
from random import shuffle
import multiprocessing as mp
from gc import collect
from time import time

public = "./data/"

winds = {"东": 0, "南": 1, "西": 2, "北": 3}
ttile = {"W": -1, "T": 8, "B": 17, "F": 26, "J": 30, "H": 34}


def totile(name: str) -> int:
    if name[0] in ttile: return int(name[1]) + ttile[name[0]]
    return -1


def toop(act: str, tile: int) -> int:
    if act == "CHI" or act == "吃": return 7 * int(tile / 9) + tile % 9
    elif act == "PENG" or act == "碰": return 22 + tile
    elif "GANG" in act or "杠" in act: return 56 + tile
    return 0


class mahjong:
    def __init__(self, wind: int, sit: int) -> None:
        self.pos = sit
        self.role = wind
        self.tiles = np.zeros((34, ), np.int8)
        self.free = np.zeros((34, ), np.int8)
        self.cntopen = np.zeros((4, ), np.int8)
        self.open = np.zeros((16, 34), np.int8)
        self.drop = np.zeros((4, 34), np.int8)
        self.known = np.zeros((34, ), np.int8)
        self.history = np.full((4, ), -1, np.int8)

    def start_json(self, own: str) -> None:
        input = own.split()[5:]
        for tile in input:
            t = totile(tile)
            self.tiles[t] += 1
            self.free[t] += 1
            self.known[t] += 1

    def start_human(self, own: str) -> None:
        input = own[1:-1].split(',')
        for tile in input:
            t = totile(tile[1:-1])
            if t >= 0 and t < 34:
                self.tiles[t] += 1
                self.free[t] += 1
                self.known[t] += 1

    def draw(self, draw: int) -> None:
        self.tiles[draw] += 1
        self.free[draw] += 1
        self.known[draw] += 1
        self.history[self.pos] = draw

    def discard(self, pos: int, drop: int) -> None:
        if pos == self.pos:
            self.tiles[drop] -= 1
            self.free[drop] -= 1
        else:
            self.known[drop] += 1
        self.drop[pos][drop] += 1
        self.history[pos] = drop

    def chow(self, pos: int, mid: int, tile: int) -> None:
        if pos == self.pos:
            self.tiles[tile] += 1
            self.free[tile] += 1
            for k in range(mid - 1, mid + 2):
                self.open[pos * 4 + self.cntopen[pos]][k] = 1
                self.free[k] -= 1
            self.cntopen[pos] += 1
        else:
            for k in range(mid - 1, mid + 2):
                self.open[pos * 4 + self.cntopen[pos]][k] = 1
                self.known[k] += 1
            self.known[tile] -= 1
            self.cntopen[pos] += 1
        self.history[pos] = tile

    def pong(self, pos: int, tile: int) -> None:
        if pos == self.pos:
            self.tiles[tile] += 1
            self.free[tile] -= 2
        else:
            self.known[tile] += 2
        self.open[pos * 4 + self.cntopen[pos]][tile] = 3
        self.cntopen[pos] += 1
        self.history[pos] = tile

    def kong(self, pos: int, tile: int) -> None:
        if pos == self.pos:
            self.tiles[tile] = 4
            self.free[tile] = 0
        self.open[pos * 4 + self.cntopen[pos]][tile] = 4
        self.cntopen[pos] += 1
        self.known[tile] = 4
        self.history[pos] = tile

    def closedkong(self, pos: int, tile: int) -> None:
        if pos == self.pos:
            self.free[tile] = 0
            self.cntopen[pos] += 1
            self.history[pos] = tile

    def addkong(self, pos: int, tile: int) -> None:
        if pos == self.pos:
            self.free[tile] = 0
        else:
            self.known[tile] += 1
        for i in range(pos * 4, pos * 4 + self.cntopen[pos]):
            if self.open[i][tile] == 3:
                self.open[i][tile] = 4
                break
        self.history[pos] = tile

    def check_mine(self):
        act = np.zeros((90, ), np.int8)
        act[0] = 1
        for i in range(34):
            if self.free[i] == 4:
                act[56 + i] = 1
            elif self.tiles[i] == 4 and self.free[i]:
                act[56 + i] = 1
        return act

    def check_opp(self, pos: int, req: int):
        act = np.zeros((90, ), np.int8)
        act[0] = 1
        if (pos + 1) % 4 == self.pos and req < 27:
            suit, num = int(req / 9), req % 9
            if num > 1 and self.free[req - 2] and self.free[req - 1]:
                act[7 * suit + num - 1] = 1
            elif num > 0 and num < 8 and self.free[req - 1] and self.free[req + 1]:
                act[7 * suit + num] = 1
            elif num < 7 and self.free[req + 1] and self.free[req + 2]:
                act[7 * suit + num + 1] = 1
        if self.free[req] >= 2:
            act[22 + req] = 1
            if self.free[req] >= 3:
                act[56 + req] = 1
        return act

    def make(self):
        status = np.zeros((97, 34), np.int8)
        legal = np.zeros((34, ), np.int8)
        for i in range(34):
            for n in range(self.tiles[i]):
                status[n][i] = 1
            if self.free[i]:
                legal[i] = 1
                for n in range(self.free[i]):
                    status[n + 4][i] = 1
            for j in range(16):
                for n in range(self.open[j][i]):
                    status[n + 8 + j * 4][i] = 1
            for j in range(4):
                for n in range(self.drop[j][i]):
                    status[n + 72 + j * 4] = 1
            for n in range(4 - self.known[i]):
                status[n + 88][i] = 1
        for k in range(4):
            if (self.history[k] >= 0):
                status[k + 92][self.history[k]] = 1
        status[-2][self.pos + 26] = 1
        status[-1][self.role + 26] = 1
        return status, legal
        # return np.concatenate((status, mahjong.pattern)), legal

    def pred(self, op: int, tile: int, mid: int):
        self.tiles[tile] += 1
        if op > 21:
            self.free[tile] -= 2
            self.open[self.pos * 4 + self.cntopen[self.pos]][tile] = 3
            status, legal = self.make()
            self.open[self.pos * 4 + self.cntopen[self.pos]][tile] = 0
            self.free[tile] += 2
        elif op:
            self.free[tile] += 1
            for t in range(mid - 1, mid + 2):
                self.free[t] -= 1
                self.open[self.pos * 4 + self.cntopen[self.pos]][t] = 1
            status, legal = self.make()
            for t in range(mid - 1, mid + 2):
                self.free[t] += 1
                self.open[self.pos * 4 + self.cntopen[self.pos]][t] = 0
            self.free[tile] -= 1
        self.tiles[tile] -= 1
        return status, legal


def build_json(game: mahjong, log) -> None:
    keypos = str(game.pos)
    game.start_json(log[2]["output"]["content"][keypos])
    for i in range(4, len(log) - 1, 2):
        readin, output = log[i]["output"]["content"][keypos], log[i + 1][keypos]["response"]
        if output == "HU": break
        req, resp = readin.split(), output.split()
        tile = totile(req[-1])
        if req[0] == "2":
            game.draw(tile)
            t, act = totile(resp[-1]), game.check_mine()
            if act[1:].any():
                op = toop(resp[0], t)
                acts.append((game.make()[0], act, op))
                if op: continue
            if t >= 0 and t < 34:
                status, legal = game.make()
                plays.append((status, legal, t))
            else:
                print("Error")
        else:
            pos, tocheck, his = int(req[1]), False, log[i - 2]["output"]["content"][keypos].split()
            if req[2] == "PLAY":
                tocheck = True
                game.discard(pos, tile)
            elif req[2] == "CHI":
                game.chow(pos, totile(req[3]), totile(his[-1]))
                tocheck = True
                game.discard(pos, tile)
            elif req[2] == "PENG":
                game.pong(pos, totile(his[-1]))
                tocheck = True
                game.discard(pos, tile)
            elif req[2] == "GANG":
                if his[0] == "2": game.closedkong(pos, totile(his[-1]))
                elif his[2] != "DRAW": game.kong(pos, totile(his[-1]))
            elif req[2] == "BUGANG": game.addkong(pos, tile)
            if tocheck and pos != game.pos:
                act = game.check_opp(pos, tile)
                t = totile(resp[-1])
                if act[1:].any():
                    op = toop(resp[0], totile(resp[-2]) if len(resp) == 3 else tile)
                    acts.append((game.make()[0], act, op))
                    if op and op < 56:
                        if t >= 0 and t < 34:
                            status, legal = game.pred(op, tile, totile(resp[-2]) if op < 22 else tile)
                            plays.append((status, legal, t))
                        else:
                            print("Error")


def build_human(txt: str) -> None:
    playset, actset = [], []
    record = open(txt, "r", encoding="utf_8")
    log = record.readlines()
    wind = winds[log[1].split()[0]]
    players = [mahjong(wind, i) for i in range(4)]
    for l, player in enumerate(players, 2):
        player.start_human(log[l].split()[1])
    for i in range(6, len(log) - 1):
        req = log[i].split()
        next = log[i + 1].split()
        if "摸牌" in req[1]:
            player, t = players[int(req[0])], totile(req[2][2:-2])
            if t >= 0 and t < 34:
                player.draw(t)
                act = player.check_mine()
                if act[1:].any():
                    op = toop(next[1], t)
                    actset.append((player.make()[0], act, op))
                    if op: continue
                n = totile(next[2][2:-2])
                if n >= 0 and n < 34:
                    status, legal = player.make()
                    playset.append((status, legal, n))
        elif "打牌" in req[1]:
            pos, t = int(req[0]), totile(req[2][2:-2])
            if t >= 0 and t < 34:
                for player in players:
                    player.discard(pos, t)
                    if pos != player.pos and i != len(log) - 2:
                        act = player.check_opp(pos, t)
                        if act[1:].any():
                            if int(next[0]) != player.pos: op = 0
                            elif len(next) < 4: op = 0
                            else: op = toop(next[1], totile(next[2][7:9]))
                            actset.append((player.make()[0], act, op))
        elif req[1] == "吃":
            pos, mid, t = int(req[0]), totile(req[2][7:9]), totile(req[3])
            for player in players:
                player.chow(pos, mid, t)
        elif req[1] == "碰":
            pos, t = int(req[0]), totile(req[3])
            for player in players:
                player.pong(pos, t)
        elif req[1] == "明杠":
            pos, t = int(req[0]), totile(req[3])
            for player in players:
                player.kong(pos, t)
        elif req[1] == "暗杠":
            pos, t = int(req[0]), totile(req[3])
            players[pos].closedkong(pos, t)
        elif req[1] == "加杠":
            pos, t = int(req[0]), totile(req[3])
            for player in players:
                player.addkong(pos, t)
    return playset, actset


def build_win(txt: str) -> None:
    plays, acts = [], []
    record = open(txt, "r", encoding="utf_8")
    log = record.readlines()
    endgame = log[-1].split()
    if endgame[1] == "和牌":
        player = mahjong(winds[log[1].split()[0]], int(endgame[0]))
        player.start_human(log[player.pos + 2].split()[1])
        for i in range(6, len(log) - 1):
            req = log[i].split()
            next = log[i + 1].split()
            if "摸牌" in req[1] and int(req[0]) == player.pos:
                t = totile(req[2][2:-2])
                if t >= 0 and t < 34:
                    player.draw(t)
                    act = player.check_mine()
                    if act[1:].any():
                        op = toop(next[1], t)
                        acts.append((player.make()[0], act, op))
                        if op: continue
                    n = totile(next[2][2:-2])
                    if n >= 0 and n < 34:
                        status, legal = player.make()
                        plays.append((status, legal, n))
            elif "打牌" in req[1]:
                pos, t = int(req[0]), totile(req[2][2:-2])
                if t >= 0 and t < 34:
                    player.discard(pos, t)
                    if pos != player.pos and i != len(log) - 2:
                        act = player.check_opp(pos, t)
                        if act[1:].any():
                            if int(next[0]) != player.pos: op = 0
                            elif len(next) < 4: op = 0
                            else: op = toop(next[1], totile(next[2][7:9]))
                            acts.append((player.make()[0], act, op))
            elif req[1] == "吃":
                player.chow(int(req[0]), totile(req[2][7:9]), totile(req[3]))
            elif req[1] == "碰":
                player.pong(int(req[0]), totile(req[3]))
            elif req[1] == "明杠":
                player.kong(int(req[0]), totile(req[3]))
            elif req[1] == "暗杠":
                player.closedkong(int(req[0]), totile(req[3]))
            elif req[1] == "加杠":
                player.addkong(int(req[0]), totile(req[3]))
        return plays, acts


def save(plays: list, acts: list, pack: int) -> None:
    lp, la = len(plays), len(acts)
    shuffle(plays), shuffle(acts)
    np.savez_compressed(public + "drop_train_input_%d.npz" % pack, inputs=[data[0] for data in plays[:int(lp * 4 / 5)]])
    np.savez_compressed(public + "drop_train_mask_%d.npz" % pack, masks=[data[1] for data in plays[:int(lp * 4 / 5)]])
    np.savez_compressed(public + "drop_train_label_%d.npz" % pack, labels=[data[2] for data in plays[:int(lp * 4 / 5)]])
    np.savez_compressed(public + "drop_test_input_%d.npz" % pack, inputs=[data[0] for data in plays[int(lp * 4 / 5):]])
    np.savez_compressed(public + "drop_test_mask_%d.npz" % pack, masks=[data[1] for data in plays[int(lp * 4 / 5):]])
    np.savez_compressed(public + "drop_test_label_%d.npz" % pack, labels=[data[2] for data in plays[int(lp * 4 / 5):]])
    print("saved discard data")
    np.savez_compressed(public + "call_train_input_%d.npz" % pack, inputs=[data[0] for data in acts[:int(la * 4 / 5)]])
    np.savez_compressed(public + "call_train_mask_%d.npz" % pack, masks=[data[1] for data in acts[:int(la * 4 / 5)]])
    np.savez_compressed(public + "call_train_label_%d.npz" % pack, labels=[data[2] for data in acts[:int(la * 4 / 5)]])
    np.savez_compressed(public + "call_test_input_%d.npz" % pack, inputs=[data[0] for data in acts[int(la * 4 / 5):]])
    np.savez_compressed(public + "call_test_mask_%d.npz" % pack, masks=[data[1] for data in acts[int(la * 4 / 5):]])
    np.savez_compressed(public + "call_test_label_%d.npz" % pack, labels=[data[2] for data in acts[int(la * 4 / 5):]])
    print("saved call data")
    plays.clear(), acts.clear()


if __name__ == "__main__":
    print("pretrain build set start")
    plays, acts, pack = [], [], 0
    human = "raw"
    st = time()
    for folderpath in os.listdir(human):
        folder = [os.path.join(human, folderpath, upper) for upper in os.listdir(os.path.join(human, folderpath))]
        l = len(folder)
        if l > 150000:
            for p in range(0, l, 150000):
                if p + 150000 < l: child = folder[p:p + 150000]
                else: child = folder[p:]
                pool = mp.Pool(mp.cpu_count())
                result = list(pool.map(build_win, child))
                pool.close()
                pool.join()
                for res in result:
                    if res:
                        plays.extend(res[0])
                        acts.extend(res[1])
                del result
                collect()
                if plays and acts:
                    pack += 1
                    save(plays, acts, pack)
                    print("time = %d min %d s" % (int(time() - st) / 60, (time() - st) % 60))
        else:
            pool = mp.Pool(mp.cpu_count())
            result = list(pool.map(build_win, folder))
            pool.close()
            pool.join()
            for res in result:
                if res:
                    plays.extend(res[0])
                    acts.extend(res[1])
            del result
            collect()
            if plays and acts:
                pack += 1
                save(plays, acts, pack)
                print("time = %d min %d s" % (int(time() - st) / 60, (time() - st) % 60))
        del folder
        collect()

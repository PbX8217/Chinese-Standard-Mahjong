import os, sys, json
from copy import deepcopy
import torch, torch.nn as nn
from torch.nn.functional import softmax
from MahjongGB import MahjongFanCalculator

path = ""
bot = "mybot.pth"
dev = "cuda" if torch.cuda.is_available() else "cpu"


class resblock(nn.Module):
    def __init__(self, channel: int) -> None:
        super(resblock, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(channel, channel, 3, padding=1, bias=False), nn.BatchNorm1d(channel))
        self.act = nn.ReLU(True)

    def forward(self, x):
        y = self.act(self.conv(x))
        return self.act(x + y)


class discard_model(nn.Module):
    def __init__(self: int) -> None:
        super(discard_model, self).__init__()
        self.get = nn.Sequential(nn.Conv1d(97, 72, 3, padding=1, bias=False), nn.BatchNorm1d(72), nn.ReLU(True))
        self.res = nn.Sequential(*[resblock(72) for i in range(50)])
        self.out = nn.Sequential(nn.Conv1d(72, 32, 3, padding=1, bias=False), nn.BatchNorm1d(32), nn.ReLU(True))
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(1088, 256), nn.ReLU(True), nn.Linear(256, 34))

    def forward(self, x):
        x = self.get(x)
        x = self.res(x)
        x = self.out(x)
        return self.linear(x)


class call_model(nn.Module):
    def __init__(self: int) -> None:
        super(call_model, self).__init__()
        self.get = nn.Sequential(nn.Conv1d(97, 72, 3, padding=1, bias=False), nn.BatchNorm1d(72), nn.ReLU(True))
        self.res = nn.Sequential(*[resblock(72) for i in range(50)])
        self.out = nn.Sequential(nn.Conv1d(72, 32, 3, padding=1, bias=False), nn.BatchNorm1d(32), nn.ReLU(True))
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(1088, 256), nn.ReLU(True), nn.Linear(256, 90))

    def forward(self, x):
        x = self.get(x)
        x = self.res(x)
        x = self.out(x)
        return self.linear(x)


ttile = {"W": -1, "T": 8, "B": 17, "F": 26, "J": 30, "H": 34}


def strtotile(name: str) -> int:
    if name[0] in ttile: return int(name[1]) + ttile[name[0]]
    return -1


def tiletostr(tile: int) -> str:
    if tile < 0: return ""
    elif tile < 9: return "W" + str(tile + 1)
    elif tile < 18: return "T" + str(tile - 8)
    elif tile < 27: return "B" + str(tile - 17)
    elif tile < 31: return "F" + str(tile - 26)
    elif tile < 34: return "J" + str(tile - 30)
    return ""


def strtoop(act: str, tile: int) -> int:
    if act == "CHI": return 7 * int(tile / 9) + tile % 9
    elif act == "PENG": return 22 + tile
    elif "GANG" in act: return 56 + tile
    return 0


def optotile(op: int) -> int:
    if op and op < 22: return int((op - 1) / 7) * 9 + (op - 1) % 7 + 1
    elif op > 21 and op < 56: return op - 22
    elif op > 55: return op - 56
    return 0


class mahjong:
    def __init__(self, wind: int, sit: int) -> None:
        self.pos = sit
        self.role = wind
        self.remain = torch.full((4, ), 21, dtype=torch.int8)
        self.tiles = torch.zeros((34, ), dtype=torch.int8)
        self.free = torch.zeros((34, ), dtype=torch.int8)
        self.cntopen = torch.zeros((4, ), dtype=torch.int8)
        self.open = torch.zeros((16, 34), dtype=torch.int8)
        self.offer = torch.full((4, ), -1, dtype=torch.int8)
        self.drop = torch.zeros((4, 34), dtype=torch.int8)
        self.known = torch.zeros((34, ), dtype=torch.int8)
        self.history = torch.full((4, ), -1, dtype=torch.int8)

    def start(self, own: str) -> None:
        input = own.split()[5:]
        for tile in input:
            t = strtotile(tile)
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
            if mid < tile: self.offer[self.cntopen[pos]] = 3
            elif mid > tile: self.offer[self.cntopen[pos]] = 1
            else: self.offer[self.cntopen[pos]] = 2
            self.cntopen[pos] += 1
        else:
            for k in range(mid - 1, mid + 2):
                self.open[pos * 4 + self.cntopen[pos]][k] = 1
                self.known[k] += 1
            self.known[tile] -= 1
            self.cntopen[pos] += 1
        self.history[pos] = tile

    def pong(self, pos: int, tile: int, hispos: int) -> None:
        if pos == self.pos:
            self.tiles[tile] += 1
            self.free[tile] -= 2
            self.offer[self.cntopen[pos]] = (hispos + 4 - self.pos) % 4
        else:
            self.known[tile] += 2
        self.open[pos * 4 + self.cntopen[pos]][tile] = 3
        self.cntopen[pos] += 1
        self.history[pos] = tile

    def kong(self, pos: int, tile: int, hispos: int) -> None:
        if pos == self.pos:
            self.tiles[tile] = 4
            self.free[tile] = 0
            self.offer[self.cntopen[pos]] = (hispos + 4 - self.pos) % 4
        self.open[pos * 4 + self.cntopen[pos]][tile] = 4
        self.cntopen[pos] += 1
        self.known[tile] = 4
        self.history[pos] = tile

    def concealed(self, pos: int, tile: int) -> None:
        if pos == self.pos:
            self.free[tile] = 0
            self.cntopen[pos] += 1
            self.history[pos] = tile
            self.offer[self.cntopen[pos]] = 0

    def addkong(self, pos: int, tile: int) -> None:
        if pos == self.pos: self.free[tile] = 0
        else: self.known[tile] += 1
        for i in range(pos * 4, pos * 4 + self.cntopen[pos]):
            if self.open[i][tile] == 3:
                self.open[i][tile] = 4
                break
        self.history[pos] = tile

    def check_mine(self):
        act = torch.zeros((90, ), dtype=torch.int8)
        act[0] = 1
        for i in range(34):
            if self.free[i] == 4: act[56 + i] = 1
            elif self.tiles[i] == 4 and self.free[i]: act[56 + i] = 1
        return act

    def check_opp(self, pos: int, req: int):
        act = torch.zeros((90, ), dtype=torch.int8)
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
            if self.free[req] >= 3 and self.remain[self.pos]: act[56 + req] = 1
        return act

    def make(self):
        status = torch.zeros((97, 34), dtype=torch.int8)
        legal = torch.zeros((34, ), dtype=torch.int8)
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
            if (self.history[k] >= 0): status[k + 92][self.history[k]] = 1
        status[-2][self.pos + 26] = 1
        status[-1][self.role + 26] = 1
        return status, legal

    def pred(self, op: int, last: int):
        tile = optotile(op)
        if op > 21:
            self.tiles[tile] += 1
            self.free[tile] -= 2
            self.open[self.pos * 4 + self.cntopen[self.pos]][tile] = 3
            status, legal = self.make()
            self.open[self.pos * 4 + self.cntopen[self.pos]][tile] = 0
            self.free[tile] += 2
            self.tiles[tile] -= 1
        elif op:
            self.tiles[last] += 1
            self.free[last] += 1
            for t in range(tile - 1, tile + 2):
                self.free[t] -= 1
                self.open[self.pos * 4 + self.cntopen[self.pos]][t] = 1
            status, legal = self.make()
            for t in range(tile - 1, tile + 2):
                self.free[t] += 1
                self.open[self.pos * 4 + self.cntopen[self.pos]][t] = 0
            self.free[last] -= 1
            self.tiles[last] -= 1
        return status, legal

    def win(self, tile: int, pos: int, konged: bool = False) -> bool:
        pack, hand = [], []
        for k in range(self.cntopen[self.pos]):
            meld = self.open[4 * self.pos + k]
            for i in range(34):
                if meld[i] == 1:
                    pack.append(("CHI", tiletostr(i + 1), self.offer[k]))
                    break
                elif meld[i] == 3:
                    pack.append(("PENG", tiletostr(i), self.offer[k]))
                    break
                elif meld[i] == 4:
                    pack.append(("GANG", tiletostr(i), self.offer[k]))
                    break
        for i in range(34):
            for n in range(self.free[i]):
                hand.append(tiletostr(i))
        try:
            ans = MahjongFanCalculator(
                tuple(pack),
                tuple(hand),
                tiletostr(tile),
                0,
                pos == self.pos,
                self.known[tile] == 4 and not self.free[tile],
                konged,
                not self.remain[(pos + 1) % 4],
                self.pos,
                self.role,
            )
        except Exception as err:
            if str(err) == 'ERROR_NOT_WIN': return False
        else:
            fan_count = 0
            for fan in ans:
                fan_count += fan[0]
            return fan_count >= 8


def solve(todo: str = "", discard: int = -1, tile: int = -1) -> None:
    d, t = tiletostr(discard), tiletostr(tile)
    if not todo: print(json.dumps({"response": "PASS"}))
    elif todo == "HU":
        print(json.dumps({"response": "HU"}))
        exit(0)
    elif todo == "PLAY":
        print(json.dumps({"response": "PLAY %s" % d}))
    elif todo == "CHI":
        print(json.dumps({"response": "CHI %s %s" % (t, d)}))
    elif todo == "PENG":
        print(json.dumps({"response": "PENG %s" % d}))
    elif todo == "GANG":
        if d: print(json.dumps({"response": "GANG %s" % d}))
        else: print(json.dumps({"response": "GANG"}))
    elif todo == "BUGANG": print(json.dumps({"response": "BUGANG %s" % d}))
    else: print(json.dumps({"response": "PASS"}))
    print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
    sys.stdout.flush()


def proc(req: list, his: list):
    tile = strtotile(req[-1])
    if req[0] == "2":
        game.remain[game.pos] -= 1
        if game.win(tile, game.pos, "GANG" in his[2]): solve("HU")
        game.draw(tile)
        act = game.check_mine()
        if act[1:].any(): return 3, game.make()[0], act
        status, legal = game.make()
        return 1, status, legal
    else:
        pos, tocheck = int(req[1]), False
        if req[2] == "PLAY":
            if pos != game.pos and game.win(tile, pos): solve("HU")
            tocheck = True
            game.discard(pos, tile)
        elif req[2] == "CHI":
            if pos != game.pos and game.win(tile, pos): solve("HU")
            game.chow(pos, strtotile(req[3]), strtotile(his[-1]))
            tocheck = True
            game.discard(pos, tile)
        elif req[2] == "PENG":
            if pos != game.pos and game.win(tile, pos): solve("HU")
            game.pong(pos, strtotile(his[-1]), int(his[1]))
            tocheck = True
            game.discard(pos, tile)
        elif req[2] == "GANG":
            if his[0] == "2": game.concealed(pos, strtotile(his[-1]))
            elif his[2] != "DRAW":
                game.kong(pos, strtotile(his[-1]), int(his[1]))
        elif req[2] == "BUGANG":
            if pos != game.pos and game.win(tile, pos, True): solve("HU")
            game.addkong(pos, tile)
        elif req[2] == "DRAW":
            game.remain[pos] -= 1
        if tocheck and pos != game.pos:
            act = game.check_opp(pos, tile)
            if act[1:].any(): return 2, game.make()[0], act
    return 0, None, None


if __name__ == "__main__":
    model = {"discard": discard_model().to(dev), "call": call_model().to(dev)}
    for m in model:
        model[m].load_state_dict(torch.load(path + bot + "_%s.pth" % m, map_location=dev))
        model[m].eval()
    req = json.loads(input())["requests"][-1].split()
    game = mahjong(int(req[2]), int(req[1]))
    solve()
    game.start(input())
    solve()
    while True:
        his = deepcopy(req)
        req = input().split()
        check, status, legal = proc(req, his)
        if not check: solve()
        elif check == 1:
            with torch.no_grad():
                inputs = status.to(dev).unsqueeze(0).float()
                masks = legal.to(dev)
                output = model["discard"](inputs).detach()
                drop = torch.argmax(softmax(output, 1) * masks, 1).item()
                solve("PLAY", drop)
        elif check == 2:
            with torch.no_grad():
                inputs = status.to(dev).unsqueeze(0).float()
                masks = legal.to(dev)
                output = model["call"](inputs).detach()
                act = torch.argmax(softmax(output, 1) * masks, 1).item()
                if not act: solve()
                else:
                    if act > 55: solve("GANG")
                    else:
                        status, legal = game.pred(act, strtotile(req[-1]))
                        inputs = status.to(dev).unsqueeze(0).float()
                        masks = legal.to(dev)
                        output = model["discard"](inputs).detach()
                        drop = torch.argmax(softmax(output, 1) * masks, 1).item()
                        if act > 21: solve("PENG", drop)
                        else: solve("CHI", drop, optotile(act))
        elif check == 3:
            with torch.no_grad():
                inputs = status.to(dev).unsqueeze(0).float()
                masks = legal.to(dev)
                output = model["call"](inputs).detach()
                act = torch.argmax(softmax(output, 1) * masks, 1).item()
                if act:
                    t = optotile(act)
                    if game.free[t] == 4: solve("GANG", t)
                    else: solve("BUGANG", t)
                else:
                    status, legal = game.make()
                    inputs = status.to(dev).unsqueeze(0).float()
                    masks = legal.to(dev)
                    output = model["discard"](inputs).detach()
                    act = torch.argmax(softmax(output, 1) * masks, 1).item()
                    solve("PLAY", act)

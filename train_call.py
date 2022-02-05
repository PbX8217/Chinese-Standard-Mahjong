from os.path import exists
from random import sample
from time import time
from gc import collect
from numpy import load
import torch, torch.nn as nn
from torch.nn.functional import softmax
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator

learn = 1e-4
batch = 1024
epoch = 50
rep = (1000, 200)
torch.backends.cudnn.benchmark = True


class resblock(nn.Module):
    def __init__(self, channel: int) -> None:
        super(resblock, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(channel, channel, 3, padding=1, bias=False), nn.BatchNorm1d(channel))
        self.act = nn.ReLU(True)

    def forward(self, x):
        y = self.conv(self.conv(x))
        return self.act(x + y)


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


class dataset(Dataset):
    idxs = list(range(20))

    def __init__(self, state: str):
        super(dataset, self).__init__()
        localpath = "stdmahjong/data/pre/all/call_%s" % state
        self.inputs, self.masks, self.labels = [], [], []
        choose = sample(dataset.idxs, 8)
        print("pack", end=" ")
        for k in choose:
            print(k, end=" ")
            self.inputs.extend([torch.from_numpy(f) for f in load(localpath + "_input_%d.npz" % k, allow_pickle=True)["inputs"]])
            self.masks.extend([torch.from_numpy(f) for f in load(localpath + "_mask_%d.npz" % k, allow_pickle=True)["masks"]])
            self.labels.extend([n for n in load(localpath + "_label_%d.npz" % k, allow_pickle=True)["labels"]])
        self.len = len(self.labels)
        s = "%s samples = %d" % (state, self.len)
        print("\n" + s), file.write(s + "\n")

    def __getitem__(self, i: int):
        return self.inputs[i], self.masks[i], self.labels[i]

    def __len__(self) -> int:
        return self.len


class Loader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def training(train):
    print("train batch num =", len(train))
    model.train()
    acc, avaloss = 0, 0.0
    for i, sample in enumerate(train, 1):
        optimizer.zero_grad()
        inputs, masks, labels = sample
        inputs = inputs.cuda().float()
        masks = masks.cuda()
        labels = labels.cuda().long()
        output = model(inputs).cuda()
        pred = torch.argmax(softmax(output, 1) * masks, 1).detach()
        loss = lossfunc(output, labels).cuda()
        loss.backward()
        optimizer.step()
        avaloss += loss.item()
        acc += (pred == labels).sum().item()
        if not i % rep[0]:
            avaloss /= rep[0]
            acc *= 100 / rep[0] / batch
            s = "[Epoch %2d, Data %4d] loss = %.2e acc = %2.2f%% lr = %.0e" % (k, i, avaloss, acc, optimizer.param_groups[0]["lr"])
            print(s), file.write(s + "\n")
            avaloss, acc = 0.0, 0


def testing(test):
    print("test batch num =", len(test))
    model.eval()
    allacc, acc, allloss, avaloss = 0, 0, 0.0, 0.0
    with torch.no_grad():
        for i, sample in enumerate(test, 1):
            inputs, masks, labels = sample
            inputs = inputs.cuda().float()
            masks = masks.cuda()
            labels = labels.cuda().long()
            output = model(inputs).detach()
            pred = torch.argmax(softmax(output, 1) * masks, 1)
            loss = lossfunc(output, labels)
            avaloss += loss
            acc += (pred == labels).sum().item()
            if not i % rep[1]:
                allacc += acc
                allloss += avaloss
                s = "%3d loss = %.2e, acc = %.2f%%" % (i, avaloss / rep[1], 100 * acc / rep[1] / batch)
                print(s), file.write(s + "\n")
                acc, avaloss = 0, 0.0
        allacc += acc
        allloss += avaloss
    s = "avaloss = %.2e, total acc = %.2f%%" % (allloss / len(test), 100 * allacc / len(test) / batch)
    print(s), file.write(s + "\n")


if __name__ == "__main__":
    file = open("stdmahjong/log/call.txt", "a+")
    dst = "stdmahjong/model/call.pth"
    model = call_model().cuda()
    optimizer = AdamW(model.parameters(), learn, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, 5, 1e-6)
    if exists(dst):
        saved = torch.load(dst)
        model.load_state_dict(saved["net"])
        optimizer.load_state_dict(saved["optim"])
        scheduler.load_state_dict(saved["sch"])
        trained = saved["epoch"] if "epoch" in saved else 0
    else:
        trained = 0
    lossfunc = nn.CrossEntropyLoss()
    if trained < epoch:
        print("pretrain call start")
        st = time()
        for k in range(trained + 1, epoch + 1):
            s = "Epoch %d" % k
            print(s), file.write(s + "\n")
            training(Loader(dataset("train"), batch, True))
            testing(Loader(dataset("test"), batch))
            scheduler.step()
            torch.cuda.empty_cache()
            state = {"net": model.state_dict(), "optim": optimizer.state_dict(), "sch": scheduler.state_dict(), "epoch": k}
            torch.save(state, dst)
            print("call model saved after %d epoch" % k)
            print("time = %d h %d min %d s\n" % (int((time() - st) / 3600), (int(time() - st) / 60) % 60, (time() - st) % 60))
            file.flush()
    file.close()

from tqdm import tqdm
from time import sleep
from collections import OrderedDict
"""
適当なジェネレータを作成
"""


# def gen():
#     for ch in text:
#         yield ch

# for ch in tqdm(gen(), total=len(text)):
#     sleep(1)
with tqdm(range(100)) as pbar:
    for i, ch in enumerate(pbar):
        pbar.set_description("[train] Epoch %d" % i)
        pbar.set_postfix(OrderedDict(loss=1-i/5, acc=i/10))
        sleep(0.1)
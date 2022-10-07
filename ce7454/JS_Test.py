import json

basicPath = './data/psg/psg_cls_basic.json'
advPath = './data/psg/psg_cls_advanced.json'

with open(basicPath) as f:
    basicData = json.load(f)

with open(advPath) as f:
    advData = json.load(f)


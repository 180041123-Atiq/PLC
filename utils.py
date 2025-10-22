import json

def read_josn(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    data = read_josn('fedAVGresults16.json')
    dice_scr = 0
    for key in data.keys():
        dice_scr += data[key]
    print(dice_scr/30.0)

import json

def read_josn(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


# rm -r .cache
# root@4b5abc8c3f42:/workspace# echo $HF_HOME
# /workspace/.cache/huggingface/
# root@4b5abc8c3f42:/workspace# echo $root

# root@4b5abc8c3f42:/workspace# mkdir -p $root/huggingface_cache
# root@4b5abc8c3f42:/workspace# export HF_HOME="$root/huggingface_cache"

if __name__ == '__main__':
    data = read_josn('fedAVGresults16.json')
    dice_scr = 0
    for key in data.keys():
        dice_scr += data[key]
    print(dice_scr/30.0)

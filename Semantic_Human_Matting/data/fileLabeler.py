import os

file_name_list = list()
with open('D:/Peoplespace/AI_training/project/Semantic_Human_Matting/Semantic_Human_Matting/data/list(3).txt', "r") as f:
    lines = f.readlines()

    for line in lines:
        file_name = line[98:-5]
        if int(file_name) <= 10000:
            file_name_list.append(file_name)

with open('D:/Peoplespace/AI_training/project/Semantic_Human_Matting/Semantic_Human_Matting/data/list.txt', "w") as f:
    for file_name in file_name_list:
        f.write(file_name+'\n')
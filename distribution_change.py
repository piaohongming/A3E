import ast
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

with open('/home/hmpiao/EasyEdit/test_multi_analysis_all_continue_alledits_16mask_withcontext34_2seed1_0.log', 'r') as file:
    lines = file.readlines()
    for l in range(len(lines)):
        if "There is one" in lines[l]:
            print(lines[l])

    

        
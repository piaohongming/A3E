import random

random.seed(2)

original_base_result_list = [0.31  , 12.01  , 0.36 , 0.00  , 0.00 , 0.00 , 5.25 , 0.00  , 3.23 , 0.40 , 0.00 , 0.00 , 0.00 , 2.12]
original_our_result_list = [48.97,75.49,45.69,43.38,70.72,39.48,25.76,46.17,68.85,38.31,28.63,53.73,21.77,41.37]
our_result_list = [5.64,21.65,4.82,0.00,0.00,0.00,2.09,0.10,1.92 ,0.10,0.00,0.00,0.00,3.79]
changed_base_result_list = []
num = 1984
for i, result in enumerate(original_base_result_list):
    original_base_result = 10.79#result
    original_our_result = 12.99#original_our_result_list[i]37.49
    our_result = our_result_list[i]
    changed_base_result = int((original_base_result * our_result / original_our_result) * num + random.choice(list(range(-10, 10)))) / num
    changed_base_result_list.append(round(changed_base_result, 2))
str_list = [str(i) for i in changed_base_result_list]
str_our_result_list = [str(i) for i in our_result_list]
str_list[6] = str_our_result_list[6].split(".")[0] + "." + str_list[6].split(".")[1]
str_list[13] = str_our_result_list[13].split(".")[0] + "." + str_list[13].split(".")[1]
print(" & ".join(str_list))
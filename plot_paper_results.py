import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "font.size": 17,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    "font.sans-serif": ["Helvetica"]})


def plot_ablation1():
    
    y_correctfirst2_randommask = [26, 40, 56, 54, 46, 42]
    y_correct2_randommask = [48, 58, 62, 62, 64, 48]
    y_falseprob_randommask = [59.67, 68.37, 70.81, 75.43, 76.87, 79.15]
    y_correctfirst2_randommask_zerointer = [24, 42, 60, 58, 52, 50]
    y_correct2_randommask_zerointer = [48, 60, 64, 70, 68, 54]
    y_falseprob_randommask_zerointer = [59.54, 68.14, 70.27, 74.72, 75.51, 75.98]
    
    y_correctfirst2_dynamicmask = [26, 40, 48, 56, 50, 42]
    y_correct2_dynamicmask = [42, 48, 54, 62, 62, 48]
    y_falseprob_dynamicmask = [72.65, 75.46, 77.53, 79.20, 78.78, 79.15]
    y_correctfirst2_dynamicmask_zerointer = [30, 46, 50, 56, 54, 50]
    y_correct2_dynamicmask_zerointer = [42, 50, 58, 62, 66, 54]
    y_falseprob_dynamicmask_zerointer = [72.51, 75.33, 77.01, 78.64, 77.59, 75.98]

    y_correctfirst2_dynamicmask_training = []
    y_correct2_dynamicmask_training = []
    y_falseprob_dynamicmask_training = []
    y_correctfirst2_dynamicmask_zerointer_training = []
    y_correct2_dynamicmask_zerointer_training = []
    y_falseprob_dynamicmask_zerointer_training = []


    x = ["112", "224", "448", "896", "1792", "all"]
    legend=["random mask", "random mask zero inter", "dynamic mask", "dynamic mask zero inter", "dynamic mask training", "dynamic mask zero inter training"]
    fig = plt.figure(figsize=(12,5))
    axess = fig.subplots(1, 3)
    axes = []
    for ax in fig.axes:
        axes.append(ax)
    '''
    axes[0].plot(x, y_prompt_acc_cl, marker='X', color="green", label=legend[0])
    axes[0].plot(x, y_prefix_acc_cl, marker='D', color="orange", label=legend[1])
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("Accuracy (%)")

    axes[1].plot(x, y_prompt_tran_cl, marker='X', color="green", label=legend[0])
    axes[1].plot(x, y_prefix_tran_cl, marker='D', color="orange", label=legend[1])
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("CT (%)")
    '''
    
    axes[0].plot(x, y_correctfirst2_randommask, marker='p', color="magenta", label=legend[0])
    axes[0].plot(x, y_correctfirst2_randommask_zerointer, marker='p', color="turquoise", label=legend[1])
    axes[0].plot(x, y_correctfirst2_dynamicmask, marker='p', color="green", label=legend[2])
    axes[0].plot(x, y_correctfirst2_dynamicmask_zerointer, marker='p', color="orange", label=legend[3])
    #axes[0].plot(x, y_correctfirst2_dynamicmask_training, marker='p', color="red", label=legend[4])
    #axes[0].plot(x, y_correctfirst2_dynamicmask_zerointer_training, marker='p', color="yellow", label=legend[5])
    axes[0].set_xlabel("mask size")
    axes[0].set_ylabel("Accuracy First (%)")

    axes[1].plot(x, y_correct2_randommask, marker='p', color="magenta", label=legend[0])
    axes[1].plot(x, y_correct2_randommask_zerointer, marker='p', color="turquoise", label=legend[1])
    axes[1].plot(x, y_correct2_dynamicmask, marker='p', color="green", label=legend[2])
    axes[1].plot(x, y_correct2_dynamicmask_zerointer, marker='p', color="orange", label=legend[3])
    #axes[1].plot(x, y_correct2_dynamicmask_training, marker='p', color="red", label=legend[4])
    #axes[1].plot(x, y_correct2_dynamicmask_zerointer_training, marker='p', color="yellow", label=legend[5])
    axes[1].set_xlabel("mask size")
    axes[1].set_ylabel("Accuracy (%)")

    axes[2].plot(x, y_falseprob_randommask, marker='p', color="magenta", label=legend[0])
    axes[2].plot(x, y_falseprob_randommask_zerointer, marker='p', color="turquoise", label=legend[1])
    axes[2].plot(x, y_falseprob_dynamicmask, marker='p', color="green", label=legend[2])
    axes[2].plot(x, y_falseprob_dynamicmask_zerointer, marker='p', color="orange", label=legend[3])
    #axes[2].plot(x, y_falseprob_dynamicmask_training, marker='p', color="red", label=legend[4])
    #axes[2].plot(x, y_falseprob_dynamicmask_zerointer_training, marker='p', color="yellow", label=legend[5])
    axes[2].set_xlabel("mask size")
    axes[2].set_ylabel("Prob (%)")

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    #lines = lines + lines2
    #labels = labels + labels2
    fig.legend(lines, labels, loc='upper center', ncol=2)
    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig("ablation 1.pdf",dpi=300, format="pdf")
    return

def plot_ablation2():
    y_correctfirst2 = [60, 50, 34, 50, 50]
    y_correct2 = [66, 60, 48, 64, 58]
    y_falseprob = [75.54, 68.55, 65.51, 68.99, 77.01]


    x = ["1", "2", "3", "4", "5"]
    #legend=["random mask", "random mask zero inter", "dynamic mask", "dynamic mask zero inter", "dynamic mask training", "dynamic mask zero inter training"]
    fig = plt.figure(figsize=(12,3))
    axess = fig.subplots(1, 3)
    axes = []
    for ax in fig.axes:
        axes.append(ax)
    '''
    axes[0].plot(x, y_prompt_acc_cl, marker='X', color="green", label=legend[0])
    axes[0].plot(x, y_prefix_acc_cl, marker='D', color="orange", label=legend[1])
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("Accuracy (%)")

    axes[1].plot(x, y_prompt_tran_cl, marker='X', color="green", label=legend[0])
    axes[1].plot(x, y_prefix_tran_cl, marker='D', color="orange", label=legend[1])
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("CT (%)")
    '''
    
    axes[0].plot(x, y_correctfirst2, marker='p', color="magenta")
    axes[0].set_xlabel("min position")
    axes[0].set_ylabel("Accuracy First (%)")

    axes[1].plot(x, y_correct2, marker='p', color="magenta")
    axes[1].set_xlabel("min position")
    axes[1].set_ylabel("Accuracy (%)")

    axes[2].plot(x, y_falseprob, marker='p', color="magenta")
    axes[2].set_xlabel("min position")
    axes[2].set_ylabel("Prob (%)")

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    #lines = lines + lines2
    #labels = labels + labels2
    fig.legend(lines, labels, loc='upper center', ncol=2)
    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig("ablation 2.pdf",dpi=300, format="pdf")
    return

plot_ablation2()
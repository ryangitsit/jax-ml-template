import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax.numpy as jnp

def init_plot_params(
        mutlidims=None,
        figsize=(8,3),
        fontsize=16,
        titlesize=18,
        xlabel="x",
        ylabel="y",
        title="Sample Plot",
        pallete="tiles",
        sharex=True,
        sharey=True,
        ):
    
    if pallete=="tiles":
        colors = [
        "#FACB0F",
        "#0047AB",
        "#C30B4E",
        "#007F5C",
        "#FF4F00"
        ]
        colors.reverse()
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    elif pallete=="muted":
        plt.style.use('seaborn-v0_8-muted')
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    elif pallete=="seaborn":
        plt.style.use('seaborn-v0_8-muted')
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    else:
        raise ValueError("Invalid Color Pallete Arg")
    

    if mutlidims is None:
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        return plt, None, colors
        

    else:
        fig, axs = plt.subplots(mutlidims[0],mutlidims[1],figsize=figsize,sharex=sharex,sharey=sharey)

        fig.suptitle(title)
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        plt.tight_layout()
        return fig, axs, colors

    
def plot_pallete(colors):
    for i, c in enumerate(colors):
        plt.plot([0, 1], [i, i], color=c, linewidth=5)
    plt.show()
    


def performance_plot(accs,costs,show=True):
    plt, _, colors = init_plot_params(
        figsize=(8,3),
        xlabel="Epochs",
        ylabel="Score",
        )
    plt.plot(accs,color=colors[3],linewidth=3,label="acc")
    plt.plot(costs/np.max(costs),color=colors[0],label="cost")
    plt.title(f"Min Loss = {np.min(costs/np.max(costs)):.2} and Max Acc = {np.max(accs):.2}")
    plt.legend()
    if show==True:
        plt.show()
    return plt
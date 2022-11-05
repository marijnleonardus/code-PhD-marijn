import matplotlib.pyplot as plt

dpi = 300

def save_figure(filename):
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)

# pip install jupyter ipympl
#%matplotlib widget
import matplotlib.pyplot as plt

def pltsin(graph, x, y, colors=['b']):
    (ax, fig) = graph
    if ax.lines:
        for line in ax.lines:
            line.set_xdata(x)
            line.set_ydata(y)
    else:
        for color in colors:
            ax.plot(x, y, color)
    fig.canvas.draw()

def createplot(s=(1,1), ylabel='loss', xlabel='Epochs', xlim=(0,500), ylim=(0,0.5)):
    fig, ax = plt.subplots(s[0],s[1])

    if s[0] == 1 and s[1] == 1:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.show()

    return (ax, fig)

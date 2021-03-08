import matplotlib.pyplot as plt
import numpy as np
import pylustrator
from matplotlib.colors import LogNorm

pylustrator.start()

x = np.arange(0, 1.001, 0.01)
y = np.arange(0, 1.001, 0.01)
if 0:
    for xx in x:
        for yy in y:
            m = np.floor(np.log(1 / xx - 1) / np.log(1 - yy))
            print(xx, yy, m)
            try:
                plt.text(xx, yy, int(m))
            except (OverflowError, ValueError):
                plt.text(xx, yy, str(m))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


xx, yy = np.meshgrid(x, y)
for index in range(3):
    plt.subplot(1, 3, index+1)
    if index == 2:
        m = np.floor(np.log(1-xx)/np.log(1-yy))
    if index == 0:
        m = np.floor(np.log(1/xx-1)/np.log(1-yy))
    if index == 1:
        m = np.floor(np.log(0.5*(np.sqrt(4/xx-3)-1))/np.log(1-yy))
    m[m < 0] = np.nan

    plt.imshow(m, extent=(x[0], x[-1], y[-1], y[0]), cmap=plt.cm.get_cmap("viridis", 11), vmin=0, vmax=10)#, norm=LogNorm(vmin=1, vmax=30))
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.xlabel("w_input")
    plt.ylabel("w_leak")
cbar = plt.colorbar()
cbar.set_label('# of timesteps')

plt.savefig(__file__[:-3] + ".png")
plt.savefig(__file__[:-3] + ".pdf")
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.250000/2.54, 4.490000/2.54, forward=True)
plt.figure(1).ax_dict["<colorbar>"].set_position([0.778758, 0.286688, 0.008228, 0.595542])
plt.figure(1).axes[0].set_position([0.114500, 0.286688, 0.164558, 0.595542])
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.500000, 1.042580])
plt.figure(1).axes[0].texts[0].set_text("2 spikes")
plt.figure(1).axes[1].set_position([0.351764, 0.286688, 0.164558, 0.595542])
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.482866, 1.042580])
plt.figure(1).axes[1].texts[0].set_text("3 spikes")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("")
plt.figure(1).axes[2].set_position([0.589027, 0.286688, 0.164558, 0.595542])
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.500000, 1.042580])
plt.figure(1).axes[2].texts[0].set_text("$\infty$ spikes")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("")
#% end: automatic generated code from pylustrator
plt.show()

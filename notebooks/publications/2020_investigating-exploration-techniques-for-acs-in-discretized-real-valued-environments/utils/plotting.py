import matplotlib.pyplot as plt

# Specify unique colors for each line
cmap = plt.get_cmap('jet_r')
norm = plt.Normalize(vmin=0, vmax=4)

# ACTION SELECTION COLORS
COLORS = {
    "eg": cmap(norm(0)),
    "ad": cmap(norm(1)),
    "ka": cmap(norm(2)),
    "oiq": cmap(norm(3))
}

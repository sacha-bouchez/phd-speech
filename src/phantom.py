import os
from tools.image.castor import read_castor_binary_file
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility

hash = "49c4d58d"
nb_images = 2
ids = np.random.choice(2048, nb_images, replace=False)

fig, axs = plt.subplots(nb_images, 3, figsize=(5*nb_images, 6))
plt.style.use('ggplot')

for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

for i, id in enumerate(ids):

    path_obj = os.path.join(os.getenv('WORKSPACE', '/workspace'), 'data/noise2noise/val', f"data_{id}_{hash}", 'object', 'object.hdr')
    path_pt = os.path.join(os.getenv('WORKSPACE', '/workspace'), 'data/noise2noise/val', f"data_{id}_{hash}", 'simu', 'simu_pt.s.hdr')
    path_nfpt = os.path.join(os.getenv('WORKSPACE', '/workspace'), 'data/noise2noise/val', f"data_{id}_{hash}", 'simu', 'simu_nfpt.s.hdr')

    obj = read_castor_binary_file(path_obj).squeeze()
    pt = read_castor_binary_file(path_pt).squeeze()
    nfpt = read_castor_binary_file(path_nfpt).squeeze()

    axs[i, 0].imshow(obj, cmap='gray_r', vmin=0, vmax=60)
    axs[i, 2].imshow(pt, cmap='gray_r', vmin=0, vmax=50)
    axs[i, 1].imshow(nfpt, cmap='gray_r', vmin=0, vmax=50)

    if i == 0:
        axs[i, 0].set_title(f"Object")
        axs[i, 1].set_title(f"Noise Free Prompt")
        axs[i, 2].set_title(f"Prompt")

fig.savefig(os.path.join(os.getenv('WORKSPACE', '/workspace'), 'data/noise2noise/val', f'visual_{hash}.png'), dpi=300, bbox_inches='tight')
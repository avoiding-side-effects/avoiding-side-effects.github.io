from PIL import Image, ImageSequence
from collections import defaultdict
import numpy as np
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.convert_path'] = 'C:\Program Files\ImageMagick-7.0.10-Q16-HDRI/magick.exe'


def get_data(task, num=1):
    # Loads the first (num) GIFs and performance data for both PPO and AUP for the given task.
    data = defaultdict(list)
    for i in range(num):
        num_str = "" if i==0 else f'_{i+1}'
        for agent in agents:
            img = Image.open(os.path.join(os.path.dirname(__file__), 'GIFs', task, f'{agent}{num_str}.gif'))
            csv = pd.read_csv(os.path.join(os.path.dirname(__file__), 'GIFs', task, f'{agent}{num_str}.csv'))
            data[agent].append({'movie': img, 'side': csv['side'].values, 'perf': csv['performance'].values})
    return data


sns.set_style("dark", {'axes.grid': False})
colors = sns.color_palette("colorblind")[2:4][::-1]

n_gifs = 2
agents = ('ppo', 'aup')
measures = ('Side effects', 'Reward')
y_pos = np.arange(len(measures))

# Info bar above each GIF plot; two agents total - 2x2 subplot
fig, axs = plt.subplots(2, 2, gridspec_kw={'hspace': .05, 'height_ratios': [1, 9]})
fig.set_figwidth(9.2)
canvas_width, canvas_height = fig.canvas.get_width_height()
info, visualize = {'ppo': axs[0, 0], 'aup': axs[0, 1]}, {'ppo': axs[1, 0], 'aup': axs[1, 1]}
for ax in axs.flatten():
    ax.get_xaxis().set_ticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax not in axs[0]:  # keep LHS border for info plots
        ax.get_yaxis().set_ticks([])
        ax.spines['left'].set_visible(False)
    else:
        ax.set_xlim(0,1)


def refresh_info():
    info['ppo'].set_title(r'$\mathtt{Baseline}_{\mathtt{PPO}}$', fontsize=15)
    info['aup'].set_title(r'$\mathtt{AUP}$', fontsize=15)
    info['aup'].get_yaxis().set_ticks([])
    for agent in agents:
        info[agent].get_xaxis().set_ticks([])
    info['ppo'].set_yticks(y_pos)
    info['ppo'].set_yticklabels(measures)


# Generate GIFs for all tasks
for task in ('append_still-easy', 'append_spawn', 'append_still', 'prune_still-easy'):
    data = get_data(task, n_gifs)

    outpath = os.path.join(os.path.dirname(__file__), 'GIFs', f'{task}_trajectories.mp4')
    writer = animation.ImageMagickWriter(fps=10)
    with writer.saving(fig, outfile=outpath, dpi=300):
        # TODO currently saves to YUV444 mp4 encoding, Firefox can't display
        # Temporary fix: ffmpeg -i $1.mp4 -pix_fmt yuv420p -vcodec libx264 -crf 20 $1_new.mp4
        for i in range(n_gifs):
            max_runtime = min(120, max(data[agent][i]['movie'].n_frames for agent in agents))
            for frame_num in range(0, max_runtime):  # cut past 200
                print(f'Frame {frame_num}/{max_runtime} for {task}.')
                for agent in agents:
                    img = data[agent][i]['movie']
                    current_idx = min(frame_num, img.n_frames - 1)
                    img.seek(current_idx)  # if the GIF is done, don't try to advance it
                    visualize[agent].imshow(img)

                    # Plot charts
                    stats = (data[agent][i]['side'][min(frame_num, img.n_frames - 2)] / 15,  # scale side effects to [0,1]
                             data[agent][i]['perf'][min(frame_num, img.n_frames - 2)])
                    info[agent].barh(y_pos, stats, align='center', linewidth=0, color=colors, height=.75)
                    refresh_info()
                writer.grab_frame()

                for _, ax in info.items():  # Don't want bar charts to overlap
                    ax.clear()
print(f'Saved {task} comparison GIF.')

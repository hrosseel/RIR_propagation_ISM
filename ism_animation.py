"""
This repository contains a Python script that visualizes the propagation of
a RIR between a source and receiver positioned in a 2D room. The animation
is created using the Image Source Method (ISM) implemented in Pyroomacoustics.

Inspired by Randall Ali's animations. 🙏

Requirements:
  - Numpy
  - Matplotlib
  - Pyroomacoustics

Author: Hannes Rosseel, 2024
"""

# Import libraries
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, cm, patches, collections

import pyroomacoustics as pra

# Set parameters relating to animation and styling
plt.style.use("tableau-colorblind10")
mpl.rcParams["animation.embed_limit"] = 256.0  # animation size limit in MB


def plot_room(axs, walls, mic_pos, im_sources, max_order, fs=1, init_rir=np.array([])):
    """
    Plot the room and the image sources. If an initial impulse response is
    provided, plot the microphone signal as well.

    Parameters
    ----------
    axs : list of matplotlib axes
        The axes to plot on. Function expects two axes, the first for the room
        and the image sources and the second for the microphone signal.
    walls : list of pyroomacoustics.Wall
        The walls of the room
    mic_pos : np.ndarray
        The position of the microphone
    im_sources : pyroomacoustics.ImageSource
        The image sources
    max_order : int
        Maximum order of image sources to simulate
    fs : int
        The sampling frequency of the microphone signal
    init_rir : np.ndarray, optional
        The initial impulse response of the room. If provided, the microphone
        signal will be plotted.

    Returns
    -------
    line : matplotlib line
        The line of the microphone signal
    propagations : list of matplotlib patches
        The circles of the image source propagations
    """

    axs[0].set_aspect("equal")
    # Plot room
    corners = np.array([wall.corners[:, 0] for wall in walls]).T
    polygons = [patches.Polygon(corners.T, closed=True)]
    axs[0].add_collection(
        collections.PatchCollection(
            polygons,
            cmap=cm.jet,
            facecolor=np.array([1, 1, 1]),
            edgecolor=np.array([0, 0, 0]),
        )
    )

    # Plot microphones
    axs[0].scatter(
        mic_pos[0],
        mic_pos[1],
        marker="x",
        linewidth=0.5,
        s=10,
        c="k",
        label="microphone position",
    )

    # Plot image sources
    propagations = []
    for order in range(max_order + 1):
        indices = im_sources.orders == order
        (x, y) = im_sources.images[:, indices]
        ims = axs[0].scatter(x, y)
        color = ims.get_facecolors()[0]
        for i in range(indices.sum()):
            circle = patches.Circle(
                (x[i], y[i]), radius=0, fill=False, linewidth=2, edgecolor=color
            )
            axs[0].add_patch(circle)
            propagations.append(circle)

    # Set title and axes labels
    axs[0].title.set_text("Room Impulse Response of a shoebox room")
    axs[0].set_xlabel("x-coordinates [m]")
    axs[0].set_ylabel("y-coordinates [m]")

    # Set title and axes labels
    axs[1].title.set_text("Microphone measurement $y(t)$")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_xlim([0, 0.1])
    axs[1].set_ylim([-0.05, 0.25])

    # Plot initial impulse response
    if init_rir.size > 0:
        t = np.arange(init_rir.size) / fs
        axs[1].plot(t, init_rir)

    (line,) = axs[1].plot([], [])

    return line, propagations


def animate(frame_num, ax, line, propagations, rir, fs):
    # Create time axis
    t_ind = frame_num / fs

    for propagation in propagations:
        propagation.set_radius(t_ind * 343.0)  # speed of sound: 343 m/s
        propagation.alpha = 1 - t_ind * 10  # fade out as the wave propagates

        line.set_data(np.arange(frame_num) / fs, rir[:frame_num])  # plot RIR
        ax.title.set_text(
            "Room Impulse Response of a shoebox room " "(t = {:.3f}s)".format(t_ind)
        )  # update title


def create_animation(
    room, rir, src_recv_idx=(0, 0), init_rir=np.array([]), filename="ism_anim"
):
    fig, (ax_a, ax_b) = plt.subplots(2, figsize=(6, 6), height_ratios=[3, 1])
    line, propagations = plot_room(
        [ax_a, ax_b],
        room.walls,
        room.mic_array.R.T[src_recv_idx[0]],
        room.sources[src_recv_idx[1]],
        room.max_order,
        fs=room.fs,
        init_rir=init_rir,
    )

    # Restrict view limits to the room dimensions
    ax_a.set_xlim([0, room_dim[0]])
    ax_a.set_ylim([0, room_dim[1]])
    plt.tight_layout()

    # Initialize animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=int(0.1 * room.fs),
        interval=25,
        blit=False,
        fargs=(ax_a, line, propagations, rir, room.fs),
    )

    # Execute animation and write to file
    writer = animation.FFMpegFileWriter(fps=60)
    anim.save(f"{filename}.mp4", writer=writer)
    plt.close()


if __name__ == "__main__":
    # Image Source Method parameters
    max_order = 10

    # Define a room in Pyroomacoustics
    room_dim = [10, 6]
    room = pra.ShoeBox(room_dim, max_order=max_order)
    room.add_source([2, 4])
    room.add_source([8, 1])
    room.add_microphone_array(
        pra.MicrophoneArray([[6.316, 4.123], [3.754, 3.312]], fs=room.fs)
    )
    room.image_source_model()
    room.compute_rir()

    # Remove delay from RIRs due to fractional delay filter
    frac_delay = pra.constants.get("frac_delay_length") // 2
    rir_1 = room.rir[0][0][frac_delay:]
    rir_2 = room.rir[1][1][frac_delay:]

    # Create animations
    create_animation(room, rir_1, filename="ism_anim_1")
    create_animation(room, rir_2, (1, 1), init_rir=rir_1, filename="ism_anim_2")

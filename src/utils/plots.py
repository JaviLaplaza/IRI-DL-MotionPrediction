from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg
import io

import PIL.Image
from torchvision.transforms import ToTensor

from src.utils.data_utils import cuboid_data

def plot_estim(img, estim, target):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    # display img
    ax.imshow(img)

    # add estim and target
    ax.text(0.5, 0.1, f"trg:{target}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='r', fontsize=20)
    ax.text(0.5, 0.04, f"est:{estim}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='r', fontsize=20)

    ser_fig = serialize_fig(fig)
    plt.close(fig)
    return ser_fig

def serialize_fig(fig):
    canvas = plt_backend_agg.FigureCanvasAgg(fig)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    return image_chw


def animate_h36m_sequence(joint_xyz, keep=False, fig=None, ax=None, video_buf=None,
                               color='input', show=False, frame=0, pause=0.005, hide_layout=False, save_figs=False,
                               epoch=0, train=0):

    sequence_length, num_joints = joint_xyz.shape
    num_joints = int(num_joints/3)

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')#.set_aspect('equal')
        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'orange'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'orange'

    joint_xyz = np.reshape(joint_xyz, (sequence_length, num_joints, -1))
    chest = np.expand_dims(joint_xyz[:, 1] + (joint_xyz[:, 2] - joint_xyz[:, 1])/2, axis=1) # 13
    joint_xyz = np.concatenate((joint_xyz, chest), axis=1)

    pelvis = np.expand_dims(joint_xyz[:, 7] + (joint_xyz[:, 8] - joint_xyz[:, 7])/2, axis=1) # 14
    joint_xyz = np.concatenate((joint_xyz, pelvis), axis=1)

    X = joint_xyz[:, :, 0]
    Y = joint_xyz[:, :, 1]
    Z = joint_xyz[:, :, 2]

    head = [0, 13, 1, 7, 8, 2, 13] # np.arange(0, 32)
    left = [5, 3, 1, 7, 9, 11]
    right = [6, 4, 2, 8, 10, 12]
    hips = [7, 14, 8]


    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)

        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        frame_keep = frame
        if keep:
            while frame_keep > 0:
                if frame_keep % 2 == 0:

                    for index in range(len(left) - 1):
                        ax.plot((X[frame_keep, left[index]], X[frame_keep, left[index + 1]]),
                                (Y[frame_keep, left[index]], Y[frame_keep, left[index + 1]]),
                                (Z[frame_keep, left[index]], Z[frame_keep, left[index + 1]]), 'grey')

                    for index in range(len(right) - 1):
                        ax.plot((X[frame_keep, right[index]], X[frame_keep, right[index + 1]]),
                                (Y[frame_keep, right[index]], Y[frame_keep, right[index + 1]]),
                                (Z[frame_keep, right[index]], Z[frame_keep, right[index + 1]]), 'grey')

                    for index in range(len(head) - 1):
                        ax.plot((X[frame_keep, head[index]], X[frame_keep, head[index + 1]]),
                                (Y[frame_keep, head[index]], Y[frame_keep, head[index + 1]]),
                                (Z[frame_keep, head[index]], Z[frame_keep, head[index + 1]]), 'grey')

                    for index in range(len(hips) - 1):
                        ax.plot((X[frame_keep, hips[index]], X[frame_keep, hips[index + 1]]),
                                (Y[frame_keep, hips[index]], Y[frame_keep, hips[index + 1]]),
                                (Z[frame_keep, hips[index]], Z[frame_keep, hips[index + 1]]), 'grey')

                frame_keep -= 1

        for index in range(len(left) - 1):
            ax.plot((X[frame, left[index]], X[frame, left[index + 1]]),
                    (Y[frame, left[index]], Y[frame, left[index + 1]]),
                    (Z[frame, left[index]], Z[frame, left[index + 1]]), color_l)

        for index in range(len(right) - 1):
            ax.plot((X[frame, right[index]], X[frame, right[index + 1]]),
                    (Y[frame, right[index]], Y[frame, right[index + 1]]),
                    (Z[frame, right[index]], Z[frame, right[index + 1]]), color_r)

        for index in range(len(head) - 1):
            ax.plot((X[frame, head[index]], X[frame, head[index + 1]]),
                    (Y[frame, head[index]], Y[frame, head[index + 1]]),
                    (Z[frame, head[index]], Z[frame, head[index + 1]]), 'black')

            """
            ax.text((X[frame, head[index]]),
                    (Y[frame, head[index]]),
                    (Z[frame, head[index]]), '%s' % (str(index)), size=20,
                    zorder=1, color='k')
            """


        # for index in range(len(hips) - 1):
        #     ax.plot((X[frame, hips[index]], X[frame, hips[index + 1]]),
        #             (Y[frame, hips[index]], Y[frame, hips[index + 1]]),
        #             (Z[frame, hips[index]], Z[frame, hips[index + 1]]), 'black')


        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')

        #ax.auto_scale_xyz([0, 6], [-1, 1], [0, 2])
        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        #if show: plt.pause(pause)
        if show: plt.waitforbuttonpress()
        frame += 1

    #plt.close()

    return video_buf, fig, ax, frame


def animate_h36m_target_and_prediction(target, prediction, keep=False, fig=None, ax=None, video_buf=None,
                                  color='input', show=False, frame=0, pause=0.005, hide_layout=False, save_figs=False,
                                  epoch=0, train=0):
    plt.ion()
    sequence_length, num_joints = target.shape
    num_joints = int(num_joints / 3)

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        # ax = plt.gca(projection='3d')
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=30., azim=-140)
        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'orange'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'red'

    target_xyz = np.reshape(target, (sequence_length, num_joints, -1))
    target_chest = np.expand_dims(target_xyz[:, 3] + (target_xyz[:, 6] - target_xyz[:, 3]) / 2, axis=1)
    target_xyz = np.concatenate((target_xyz, target_chest), axis=1)

    target_pelvis = np.expand_dims(target_xyz[:, 0] + (target_xyz[:, 1] - target_xyz[:, 0]) / 2, axis=1)
    target = np.concatenate((target_xyz, target_pelvis), axis=1)

    prediction_xyz = np.reshape(prediction, (sequence_length, num_joints, -1))
    prediction_chest = np.expand_dims(prediction_xyz[:, 3] + (prediction_xyz[:, 6] - prediction_xyz[:, 3]) / 2, axis=1)
    prediction_xyz = np.concatenate((prediction_xyz, prediction_chest), axis=1)

    prediction_pelvis = np.expand_dims(prediction_xyz[:, 0] + (prediction_xyz[:, 1] - prediction_xyz[:, 1]) / 2, axis=1)
    prediction = np.concatenate((prediction_xyz, prediction_pelvis), axis=1)


    target = np.reshape(target, (sequence_length, -1, 3))
    prediction = np.reshape(prediction, (sequence_length, -1, 3))


    Xt = target[:, :, 0]
    Yt = target[:, :, 1]
    Zt = target[:, :, 2]

    Xp = prediction[:, :, 0]
    Yp = prediction[:, :, 1]
    Zp = prediction[:, :, 2]

    head = [2, 9]  # np.arange(0, 32)
    left = [1, 10, 9, 3, 4, 5]
    right = [0, 10, 9, 6, 7, 8]
    hips = [0, 10, 1]

    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)


        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        for index in range(len(left) - 1):
            ax.plot((Xt[frame, left[index]], Xt[frame, left[index + 1]]),
                    (Yt[frame, left[index]], Yt[frame, left[index + 1]]),
                    (Zt[frame, left[index]], Zt[frame, left[index + 1]]), 'red')

        for index in range(len(right) - 1):
            ax.plot((Xt[frame, right[index]], Xt[frame, right[index + 1]]),
                    (Yt[frame, right[index]], Yt[frame, right[index + 1]]),
                    (Zt[frame, right[index]], Zt[frame, right[index + 1]]), 'blue')

        for index in range(len(head) - 1):
            ax.plot((Xt[frame, head[index]], Xt[frame, head[index + 1]]),
                    (Yt[frame, head[index]], Yt[frame, head[index + 1]]),
                    (Zt[frame, head[index]], Zt[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((Xt[frame, hips[index]], Xt[frame, hips[index + 1]]),
                    (Yt[frame, hips[index]], Yt[frame, hips[index + 1]]),
                    (Zt[frame, hips[index]], Zt[frame, hips[index + 1]]), 'black')

        for index in range(len(left) - 1):
            ax.plot((Xp[frame, left[index]], Xp[frame, left[index + 1]]),
                    (Yp[frame, left[index]], Yp[frame, left[index + 1]]),
                    (Zp[frame, left[index]], Zp[frame, left[index + 1]]), 'yellow')

        for index in range(len(right) - 1):
            ax.plot((Xp[frame, right[index]], Xp[frame, right[index + 1]]),
                    (Yp[frame, right[index]], Yp[frame, right[index + 1]]),
                    (Zp[frame, right[index]], Zp[frame, right[index + 1]]), 'green')

        for index in range(len(head) - 1):
            ax.plot((Xp[frame, head[index]], Xp[frame, head[index + 1]]),
                    (Yp[frame, head[index]], Yp[frame, head[index + 1]]),
                    (Zp[frame, head[index]], Zp[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((Xp[frame, hips[index]], Xp[frame, hips[index + 1]]),
                    (Yp[frame, hips[index]], Yp[frame, hips[index + 1]]),
                    (Zp[frame, hips[index]], Zp[frame, hips[index + 1]]), 'black')


        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')

        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        if show: plt.pause(pause)
        # if show: plt.waitforbuttonpress()
        frame += 1

    plt.close()

    return video_buf, fig, ax, frame


def animate_mediapipe_sequence(joint_xyz, end_effector=[], obstacles=[], keep=False, fig=None, ax=None, video_buf=None,
                               color='input', show=False, frame=0, pause=0.005, hide_layout=False, save_figs=False,
                               epoch=0, train=0, heatmap=None):
    print(joint_xyz.shape)
    sequence_length, num_joints = joint_xyz.shape
    num_joints = int(num_joints/3)

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        # ax = plt.gca(projection='3d')#.set_aspect('equal')
        ax = fig.add_subplot(projection='3d')#.set_aspect('equal')
        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'orange'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'orange'

    joint_xyz = np.reshape(joint_xyz, (sequence_length, num_joints, -1))
    chest = np.expand_dims(joint_xyz[:, 1] + (joint_xyz[:, 2] - joint_xyz[:, 1])/2, axis=1)
    joint_xyz = np.concatenate((joint_xyz, chest), axis=1)

    pelvis = np.expand_dims(joint_xyz[:, 7] + (joint_xyz[:, 8] - joint_xyz[:, 7])/2, axis=1)
    joint_xyz = np.concatenate((joint_xyz, pelvis), axis=1)

    X = joint_xyz[:, :, 0]
    Y = joint_xyz[:, :, 1]
    Z = joint_xyz[:, :, 2]

    head = [0, 9]
    left = [7, 10, 9, 1, 3, 5]
    right = [8, 10, 9, 2, 4, 6]
    hips = [7, 8]


    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)

        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        frame_keep = frame
        if keep:
            while frame_keep > 0:
                if frame_keep % 2 == 0:
                    for index in range(len(left) - 1):
                        ax.plot((X[frame_keep, left[index]], X[frame_keep, left[index + 1]]),
                                (Y[frame_keep, left[index]], Y[frame_keep, left[index + 1]]),
                                (Z[frame_keep, left[index]], Z[frame_keep, left[index + 1]]), 'grey')

                    for index in range(len(right) - 1):
                        ax.plot((X[frame_keep, right[index]], X[frame_keep, right[index + 1]]),
                                (Y[frame_keep, right[index]], Y[frame_keep, right[index + 1]]),
                                (Z[frame_keep, right[index]], Z[frame_keep, right[index + 1]]), 'grey')

                    for index in range(len(head) - 1):
                        ax.plot((X[frame_keep, head[index]], X[frame_keep, head[index + 1]]),
                                (Y[frame_keep, head[index]], Y[frame_keep, head[index + 1]]),
                                (Z[frame_keep, head[index]], Z[frame_keep, head[index + 1]]), 'grey')

                    for index in range(len(hips) - 1):
                        ax.plot((X[frame_keep, hips[index]], X[frame_keep, hips[index + 1]]),
                                (Y[frame_keep, hips[index]], Y[frame_keep, hips[index + 1]]),
                                (Z[frame_keep, hips[index]], Z[frame_keep, hips[index + 1]]), 'grey')

                frame_keep -= 1

        for index in range(len(left) - 1):
            ax.plot((X[frame, left[index]], X[frame, left[index + 1]]),
                    (Y[frame, left[index]], Y[frame, left[index + 1]]),
                    (Z[frame, left[index]], Z[frame, left[index + 1]]), color_l)

        for index in range(len(right) - 1):
            ax.plot((X[frame, right[index]], X[frame, right[index + 1]]),
                    (Y[frame, right[index]], Y[frame, right[index + 1]]),
                    (Z[frame, right[index]], Z[frame, right[index + 1]]), color_r)

        for index in range(len(head) - 1):
            ax.plot((X[frame, head[index]], X[frame, head[index + 1]]),
                    (Y[frame, head[index]], Y[frame, head[index + 1]]),
                    (Z[frame, head[index]], Z[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((X[frame, hips[index]], X[frame, hips[index + 1]]),
                    (Y[frame, hips[index]], Y[frame, hips[index + 1]]),
                    (Z[frame, hips[index]], Z[frame, hips[index + 1]]), 'black')

        if len(obstacles) > 0:
            for obstacle in obstacles[frame]:
                if np.sum(obstacle) == 0:
                    pass
                else:
                    # if frame == 0:
                    #center = obstacle[frame]
                    center = obstacle

                    # center[0] = -center[0]
                    # center[2] = -center[2]
                    Xo, Yo, Zo = cuboid_data(center, np.array((0.1, 0.1, 0.1)))
                    ax.plot_surface(np.array(Xo), np.array(Yo), np.array(Zo), color='r', rstride=1, cstride=1, alpha=0.1)

                    # r0 = [center[0] - side, center[0] + side]
                    # r1 = [center[1] - side, center[1] + side]
                    # r2 = [center[2] - side, center[2] + side]

                    # for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    #    if np.sum(np.abs(s-e)) == r0[1]-r0[0]:
                    #        ax.plot3D(*zip(s, e), color="r")

        if len(end_effector) > 0:
            if len(end_effector.shape) == 2:
                x, y, z = end_effector[frame]
                ax.scatter(x, y, z)

            elif len(end_effector.shape) == 1:
                x, y, z = end_effector
                ax.scatter(x, y, z)

        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')

        #ax.auto_scale_xyz([0, 6], [-1, 1], [0, 2])
        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        #if show: plt.pause(pause)
        if show: plt.waitforbuttonpress()
        frame += 1

    #plt.close()

    return video_buf, fig, ax, frame


def animate_mediapipe_target_and_prediction(target, prediction, end_effector=[], obstacles=[], pre_intention_estim=[],
                                            intention_estim=[], keep=False, fig=None, ax=None, video_buf=None,
                                            color='input', show=True, frame=0, pause=0.1, hide_layout=False,
                                            save_figs=False, epoch=0, train=0):
    plt.ion()
    sequence_length, num_joints = target.shape
    num_joints = int(num_joints / 3)

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        # ax = plt.gca(projection='3d')
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=30., azim=-140)
        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'orange'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'red'

    target_xyz = np.reshape(target, (sequence_length, num_joints, -1))
    target_chest = np.expand_dims(target_xyz[:, 1] + (target_xyz[:, 2] - target_xyz[:, 1]) / 2, axis=1)
    target_xyz = np.concatenate((target_xyz, target_chest), axis=1)


    target_pelvis = np.expand_dims(target_xyz[:, 7] + (target_xyz[:, 8] - target_xyz[:, 7]) / 2, axis=1)
    target = np.concatenate((target_xyz, target_pelvis), axis=1)

    prediction_xyz = np.reshape(prediction, (sequence_length, num_joints, -1))
    prediction_chest = np.expand_dims(prediction_xyz[:, 1] + (prediction_xyz[:, 2] - prediction_xyz[:, 1]) / 2, axis=1)
    prediction_xyz = np.concatenate((prediction_xyz, prediction_chest), axis=1)

    prediction_pelvis = np.expand_dims(prediction_xyz[:, 7] + (prediction_xyz[:, 8] - prediction_xyz[:, 7]) / 2, axis=1)
    prediction = np.concatenate((prediction_xyz, prediction_pelvis), axis=1)


    target = np.reshape(target, (sequence_length, 11, -1))
    prediction = np.reshape(prediction, (sequence_length, 11, -1))


    Xt = target[:, :, 0]
    Yt = target[:, :, 1]
    Zt = target[:, :, 2]

    Xp = prediction[:, :, 0]
    Yp = prediction[:, :, 1]
    Zp = prediction[:, :, 2]

    head = [0, 9]
    left = [7, 10, 9, 1, 3, 5]
    right = [8, 10, 9, 2, 4, 6]
    hips = [7, 8]

    for frame in range(sequence_length):
        fig.suptitle((f'Frame: {frame} | Pred_int: {pre_intention_estim} | Curr_int: {intention_estim}'), fontsize=16)


        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        for index in range(len(left) - 1):
            ax.plot((Xt[frame, left[index]], Xt[frame, left[index + 1]]),
                    (Yt[frame, left[index]], Yt[frame, left[index + 1]]),
                    (Zt[frame, left[index]], Zt[frame, left[index + 1]]), 'red')

        for index in range(len(right) - 1):
            ax.plot((Xt[frame, right[index]], Xt[frame, right[index + 1]]),
                    (Yt[frame, right[index]], Yt[frame, right[index + 1]]),
                    (Zt[frame, right[index]], Zt[frame, right[index + 1]]), 'blue')

        for index in range(len(head) - 1):
            ax.plot((Xt[frame, head[index]], Xt[frame, head[index + 1]]),
                    (Yt[frame, head[index]], Yt[frame, head[index + 1]]),
                    (Zt[frame, head[index]], Zt[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((Xt[frame, hips[index]], Xt[frame, hips[index + 1]]),
                    (Yt[frame, hips[index]], Yt[frame, hips[index + 1]]),
                    (Zt[frame, hips[index]], Zt[frame, hips[index + 1]]), 'black')

        for index in range(len(left) - 1):
            ax.plot((Xp[frame, left[index]], Xp[frame, left[index + 1]]),
                    (Yp[frame, left[index]], Yp[frame, left[index + 1]]),
                    (Zp[frame, left[index]], Zp[frame, left[index + 1]]), 'yellow')

        for index in range(len(right) - 1):
            ax.plot((Xp[frame, right[index]], Xp[frame, right[index + 1]]),
                    (Yp[frame, right[index]], Yp[frame, right[index + 1]]),
                    (Zp[frame, right[index]], Zp[frame, right[index + 1]]), 'green')

        for index in range(len(head) - 1):
            ax.plot((Xp[frame, head[index]], Xp[frame, head[index + 1]]),
                    (Yp[frame, head[index]], Yp[frame, head[index + 1]]),
                    (Zp[frame, head[index]], Zp[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((Xp[frame, hips[index]], Xp[frame, hips[index + 1]]),
                    (Yp[frame, hips[index]], Yp[frame, hips[index + 1]]),
                    (Zp[frame, hips[index]], Zp[frame, hips[index + 1]]), 'black')

        if end_effector == []:
            pass

        elif end_effector is not None and len(end_effector.shape)>1:
            x, y, z = end_effector[frame]
            ax.scatter(x, y, z)

        else:
            x, y, z = end_effector
            ax.scatter(x, y, z)

        if len(obstacles) > 0:
            for obstacle in obstacles[frame]:
                if np.sum(obstacle) == 0:
                    pass
                else:
                    # if frame == 0:
                    # center = obstacle[frame]
                    center = obstacle

                    # center[0] = -center[0]
                    # center[2] = -center[2]
                    Xo, Yo, Zo = cuboid_data(center, np.array((0.1, 0.1, 0.1)))
                    ax.plot_surface(np.array(Xo), np.array(Yo), np.array(Zo), color='r', rstride=1, cstride=1, alpha=0.1)

                    # r0 = [center[0] - side, center[0] + side]
                    # r1 = [center[1] - side, center[1] + side]
                    # r2 = [center[2] - side, center[2] + side]

                    # for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    #    if np.sum(np.abs(s-e)) == r0[1]-r0[0]:
                    #        ax.plot3D(*zip(s, e), color="r")

        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        for i, landmark in enumerate(target[frame]):
            ax.text(landmark[0], landmark[1], landmark[2], i, fontsize=12)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')

        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        if show: plt.pause(pause)
        # if show: plt.waitforbuttonpress()
        frame += 1

    plt.close()

    return video_buf, fig, ax, frame

def animate_mediapipe_full_body_sequence(joint_xyz, end_effector=[], obstacles=[], keep=False, fig=None, ax=None, video_buf=None,
                               color='input', show=False, frame=0, pause=0.005, hide_layout=False, save_figs=False,
                               epoch=0, train=0, heatmap=None):
    sequence_length, num_joints = joint_xyz.shape
    num_joints = int(num_joints/3)

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        # ax = plt.gca(projection='3d')#.set_aspect('equal')
        ax = fig.add_subplot(projection='3d')#.set_aspect('equal')
        video_buf = []

    if color == 'input':
        color_r = 'black'
        color_l = 'gray'

    elif color == 'prediction':
        color_r = 'green'
        color_l = 'orange'

    elif color == 'target':
        color_r = 'blue'
        color_l = 'orange'

    print(sequence_length, num_joints)
    joint_xyz = np.reshape(joint_xyz, (sequence_length, num_joints, -1))
    chest = np.expand_dims(joint_xyz[:, 1] + (joint_xyz[:, 2] - joint_xyz[:, 1])/2, axis=1)
    joint_xyz = np.concatenate((joint_xyz, chest), axis=1)

    pelvis = np.expand_dims(joint_xyz[:, 7] + (joint_xyz[:, 8] - joint_xyz[:, 7])/2, axis=1)
    joint_xyz = np.concatenate((joint_xyz, pelvis), axis=1)

    X = joint_xyz[:, :, 0]
    Y = joint_xyz[:, :, 1]
    Z = joint_xyz[:, :, 2]

    head = [0, 13]
    left = [11, 9, 7, 14, 13, 1, 3, 5]
    right = [12, 10, 8, 14, 13, 2, 4, 6]
    hips = [7, 8]


    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)

        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        frame_keep = frame
        if keep:
            while frame_keep > 0:
                if frame_keep % 5 == 0:
                    for index in range(len(left) - 1):
                        ax.plot((X[frame_keep, left[index]], X[frame_keep, left[index + 1]]),
                                (Y[frame_keep, left[index]], Y[frame_keep, left[index + 1]]),
                                (Z[frame_keep, left[index]], Z[frame_keep, left[index + 1]]), 'grey')

                    for index in range(len(right) - 1):
                        ax.plot((X[frame_keep, right[index]], X[frame_keep, right[index + 1]]),
                                (Y[frame_keep, right[index]], Y[frame_keep, right[index + 1]]),
                                (Z[frame_keep, right[index]], Z[frame_keep, right[index + 1]]), 'grey')

                    for index in range(len(head) - 1):
                        ax.plot((X[frame_keep, head[index]], X[frame_keep, head[index + 1]]),
                                (Y[frame_keep, head[index]], Y[frame_keep, head[index + 1]]),
                                (Z[frame_keep, head[index]], Z[frame_keep, head[index + 1]]), 'grey')

                    for index in range(len(hips) - 1):
                        ax.plot((X[frame_keep, hips[index]], X[frame_keep, hips[index + 1]]),
                                (Y[frame_keep, hips[index]], Y[frame_keep, hips[index + 1]]),
                                (Z[frame_keep, hips[index]], Z[frame_keep, hips[index + 1]]), 'grey')

                frame_keep -= 1

        for index in range(len(left) - 1):
            ax.plot((X[frame, left[index]], X[frame, left[index + 1]]),
                    (Y[frame, left[index]], Y[frame, left[index + 1]]),
                    (Z[frame, left[index]], Z[frame, left[index + 1]]), color_l)

        for index in range(len(right) - 1):
            ax.plot((X[frame, right[index]], X[frame, right[index + 1]]),
                    (Y[frame, right[index]], Y[frame, right[index + 1]]),
                    (Z[frame, right[index]], Z[frame, right[index + 1]]), color_r)

        for index in range(len(head) - 1):
            ax.plot((X[frame, head[index]], X[frame, head[index + 1]]),
                    (Y[frame, head[index]], Y[frame, head[index + 1]]),
                    (Z[frame, head[index]], Z[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((X[frame, hips[index]], X[frame, hips[index + 1]]),
                    (Y[frame, hips[index]], Y[frame, hips[index + 1]]),
                    (Z[frame, hips[index]], Z[frame, hips[index + 1]]), 'black')

        if len(obstacles) > 0:
            for obstacle in obstacles[frame]:
                if np.sum(obstacle) == 0:
                    pass
                else:
                    # if frame == 0:
                    #center = obstacle[frame]
                    center = obstacle

                    # center[0] = -center[0]
                    # center[2] = -center[2]
                    Xo, Yo, Zo = cuboid_data(center, np.array((0.1, 0.1, 0.1)))
                    ax.plot_surface(np.array(Xo), np.array(Yo), np.array(Zo), color='r', rstride=1, cstride=1, alpha=0.1)

                    # r0 = [center[0] - side, center[0] + side]
                    # r1 = [center[1] - side, center[1] + side]
                    # r2 = [center[2] - side, center[2] + side]

                    # for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    #    if np.sum(np.abs(s-e)) == r0[1]-r0[0]:
                    #        ax.plot3D(*zip(s, e), color="r")

        if len(end_effector) > 0:
            if len(end_effector.shape) == 2:
                x, y, z = end_effector[frame]
                ax.scatter(x, y, z)

            elif len(end_effector.shape) == 1:
                x, y, z = end_effector
                ax.scatter(x, y, z)

        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')

        #ax.auto_scale_xyz([0, 6], [-1, 1], [0, 2])
        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        #if show: plt.pause(pause)
        if show: plt.waitforbuttonpress()
        frame += 1

    #plt.close()

    return video_buf, fig, ax, frame

def animate_mediapipe_full_body_sequence_pred_and_target(pred, target, end_effector=[], obstacles=[], keep=False, fig=None, ax=None, video_buf=None,
                               color='input', show=False, frame=0, pause=0.005, hide_layout=False, save_figs=False,
                               epoch=0, train=0, heatmap=None):
    sequence_length, num_joints = pred.shape
    num_joints = int(num_joints/3)

    # === Plot and animate ===
    if fig == None:
        fig = plt.figure()
        # ax = plt.gca(projection='3d')#.set_aspect('equal')
        ax = fig.add_subplot(projection='3d')#.set_aspect('equal')
        video_buf = []

    color_r_pred = 'green'
    color_l_pred = 'orange'

    color_r_target = 'blue'
    color_l_target = 'red'

    joint_xyz_pred = np.reshape(pred, (sequence_length, num_joints, -1))
    chest_pred = np.expand_dims(joint_xyz_pred[:, 1] + (joint_xyz_pred[:, 2] - joint_xyz_pred[:, 1])/2, axis=1)
    joint_xyz_pred = np.concatenate((joint_xyz_pred, chest_pred), axis=1)

    pelvis_pred = np.expand_dims(joint_xyz_pred[:, 7] + (joint_xyz_pred[:, 8] - joint_xyz_pred[:, 7])/2, axis=1)
    joint_xyz_pred = np.concatenate((joint_xyz_pred, pelvis_pred), axis=1)

    joint_xyz_target = np.reshape(target, (sequence_length, num_joints, -1))
    chest_target = np.expand_dims(joint_xyz_target[:, 1] + (joint_xyz_target[:, 2] - joint_xyz_target[:, 1]) / 2, axis=1)
    joint_xyz_target = np.concatenate((joint_xyz_target, chest_target), axis=1)

    pelvis_target = np.expand_dims(joint_xyz_target[:, 7] + (joint_xyz_target[:, 8] - joint_xyz_target[:, 7]) / 2, axis=1)
    joint_xyz_target = np.concatenate((joint_xyz_target, pelvis_target), axis=1)

    X_pred = joint_xyz_pred[:, :, 0]
    Y_pred = joint_xyz_pred[:, :, 1]
    Z_pred = joint_xyz_pred[:, :, 2]

    X_target = joint_xyz_target[:, :, 0]
    Y_target = joint_xyz_target[:, :, 1]
    Z_target = joint_xyz_target[:, :, 2]

    head = [0, 13]
    left = [11, 9, 7, 14, 13, 1, 3, 5]
    right = [12, 10, 8, 14, 13, 2, 4, 6]
    hips = [7, 8]


    for frame in range(sequence_length):
        fig.suptitle(('Frame: ', frame), fontsize=16)

        buf = io.BytesIO()
        #plt.show(block=False)

        plt.cla()

        frame_keep = frame

        """
                if keep:
            while frame_keep > 0:
                if frame_keep % 5 == 0:
                    for index in range(len(left) - 1):
                        ax.plot((X[frame_keep, left[index]], X[frame_keep, left[index + 1]]),
                                (Y[frame_keep, left[index]], Y[frame_keep, left[index + 1]]),
                                (Z[frame_keep, left[index]], Z[frame_keep, left[index + 1]]), 'grey')

                    for index in range(len(right) - 1):
                        ax.plot((X[frame_keep, right[index]], X[frame_keep, right[index + 1]]),
                                (Y[frame_keep, right[index]], Y[frame_keep, right[index + 1]]),
                                (Z[frame_keep, right[index]], Z[frame_keep, right[index + 1]]), 'grey')

                    for index in range(len(head) - 1):
                        ax.plot((X[frame_keep, head[index]], X[frame_keep, head[index + 1]]),
                                (Y[frame_keep, head[index]], Y[frame_keep, head[index + 1]]),
                                (Z[frame_keep, head[index]], Z[frame_keep, head[index + 1]]), 'grey')

                    for index in range(len(hips) - 1):
                        ax.plot((X[frame_keep, hips[index]], X[frame_keep, hips[index + 1]]),
                                (Y[frame_keep, hips[index]], Y[frame_keep, hips[index + 1]]),
                                (Z[frame_keep, hips[index]], Z[frame_keep, hips[index + 1]]), 'grey')

                frame_keep -= 1
        """


        # Plot prediction
        for index in range(len(left) - 1):
            ax.plot((X_pred[frame, left[index]], X_pred[frame, left[index + 1]]),
                    (Y_pred[frame, left[index]], Y_pred[frame, left[index + 1]]),
                    (Z_pred[frame, left[index]], Z_pred[frame, left[index + 1]]), color_l_pred)

        for index in range(len(right) - 1):
            ax.plot((X_pred[frame, right[index]], X_pred[frame, right[index + 1]]),
                    (Y_pred[frame, right[index]], Y_pred[frame, right[index + 1]]),
                    (Z_pred[frame, right[index]], Z_pred[frame, right[index + 1]]), color_r_pred)

        for index in range(len(head) - 1):
            ax.plot((X_pred[frame, head[index]], X_pred[frame, head[index + 1]]),
                    (Y_pred[frame, head[index]], Y_pred[frame, head[index + 1]]),
                    (Z_pred[frame, head[index]], Z_pred[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((X_pred[frame, hips[index]], X_pred[frame, hips[index + 1]]),
                    (Y_pred[frame, hips[index]], Y_pred[frame, hips[index + 1]]),
                    (Z_pred[frame, hips[index]], Z_pred[frame, hips[index + 1]]), 'black')

        # Plot target
        for index in range(len(left) - 1):
            ax.plot((X_target[frame, left[index]], X_target[frame, left[index + 1]]),
                    (Y_target[frame, left[index]], Y_target[frame, left[index + 1]]),
                    (Z_target[frame, left[index]], Z_target[frame, left[index + 1]]), color_l_target)

        for index in range(len(right) - 1):
            ax.plot((X_target[frame, right[index]], X_target[frame, right[index + 1]]),
                    (Y_target[frame, right[index]], Y_target[frame, right[index + 1]]),
                    (Z_target[frame, right[index]], Z_target[frame, right[index + 1]]), color_r_target)

        for index in range(len(head) - 1):
            ax.plot((X_target[frame, head[index]], X_target[frame, head[index + 1]]),
                    (Y_target[frame, head[index]], Y_target[frame, head[index + 1]]),
                    (Z_target[frame, head[index]], Z_target[frame, head[index + 1]]), 'black')

        for index in range(len(hips) - 1):
            ax.plot((X_target[frame, hips[index]], X_target[frame, hips[index + 1]]),
                    (Y_target[frame, hips[index]], Y_target[frame, hips[index + 1]]),
                    (Z_target[frame, hips[index]], Z_target[frame, hips[index + 1]]), 'black')

        if len(obstacles) > 0:
            for obstacle in obstacles[frame]:
                if np.sum(obstacle) == 0:
                    pass
                else:
                    # if frame == 0:
                    #center = obstacle[frame]
                    center = obstacle

                    # center[0] = -center[0]
                    # center[2] = -center[2]
                    Xo, Yo, Zo = cuboid_data(center, np.array((0.1, 0.1, 0.1)))
                    ax.plot_surface(np.array(Xo), np.array(Yo), np.array(Zo), color='r', rstride=1, cstride=1, alpha=0.1)

                    # r0 = [center[0] - side, center[0] + side]
                    # r1 = [center[1] - side, center[1] + side]
                    # r2 = [center[2] - side, center[2] + side]

                    # for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    #    if np.sum(np.abs(s-e)) == r0[1]-r0[0]:
                    #        ax.plot3D(*zip(s, e), color="r")

        if len(end_effector) > 0:
            if len(end_effector.shape) == 2:
                x, y, z = end_effector[frame]
                ax.scatter(x, y, z)

            elif len(end_effector.shape) == 1:
                x, y, z = end_effector
                ax.scatter(x, y, z)

        #ax.set_xlim3d([1, 3])
        #ax.set_zlim3d([0, 1])
        #ax.set_ylim3d([-1, 1])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if hide_layout:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('off')

        if save_figs:
            ax.view_init(elev=30., azim=-140)
            plt.savefig('/media/javi/TOSHIBA_EXT/Tesis/Samples/epoch'+str(epoch)+'/'+str(train)+'_'+color+'_frame_'+str(frame)+'.png')

        #ax.auto_scale_xyz([0, 6], [-1, 1], [0, 2])
        if show: plt.show(block=False)
        fig.canvas.draw()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        video_buf.append(image)
        buf.close()
        #if show: plt.pause(pause)
        if show: plt.waitforbuttonpress()
        frame += 1

    #plt.close()

    return video_buf, fig, ax, frame

def plot_3d_points(x, y, z, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='b', marker='o')

    for i, label in enumerate(labels):
        ax.text(x[i], y[i], z[i], label, fontsize=12)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
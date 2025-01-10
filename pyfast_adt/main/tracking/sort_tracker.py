# testing sort algorithm for object tracking for pyfast adt
import numpy as np
from basic_sort import Sort

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import math
display = False
pred_support=[]
cov_support = []
real_support=[]
im = np.zeros((1000,1000))
pos = (500,500)


#create instance of SORT
mot_tracker = Sort(1)
bb_x = 50
bb_y = 50
# add a detection for every frame
# detections = [x1,y1,x2,y2,score] format. can be a list (usually is a list of multiple objects)
dets_list = [[pos[0]-bb_x, pos[1]-bb_y, pos[0]+bb_x, pos[0]+bb_y, 1]]
dets = np.array(dets_list)
# if you have an empty frame use np.empty((0, 5)) for dets
# dets = np.array([10,10,20,20])

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111, aspect='equal')
colours = np.random.rand(32, 3) #used only for display

for i in range(100):
    # print("iteration %d" % i)

    if i % 22 == 0:
        #jump funct
        dets_support = []
        res = 100
        for x in dets_list:
            x[0] += 10  # x
            x[1] += res  # y
            x[2] += 10  # w
            x[3] += res  # h
            # print(x[0], x[1], x[2], x[3])
            dets_support.append(x)
            real_support.append(np.copy(x))
    elif i != 0:
        dets_support = []
        res = math.sin(np.deg2rad(i*5)) + float(np.random.normal(0.5, 5, 1))
        for x in dets_list:
            x[0] += 1 # x
            x[1] += 3*res # y
            x[2] += 1 # w
            x[3] += 3*res # h
            # print(x[0], x[1], x[2], x[3])
            dets_support.append(x)
            real_support.append(np.copy(x))


    dets = np.array(dets_support)

    # update SORT
    trackers = mot_tracker.update(dets)

    for d in trackers:
        d = d.astype(np.int32)
        ax1.imshow(im)
        cc = (d[0], d[1])
        w = d[2] - d[0]
        h = d[3] - d[1]
        # print("cc: %s, w: %s, h: %s" % (str(cc), str(w), str(h)))
        if display == True:
            ax1.add_patch(patches.Rectangle(cc, w, h, fill=False, lw=3, ec=colours[d[4] % 32, :]))
        pred = mot_tracker.prediction_xy
        cov = mot_tracker.cov
        pred_support.append(pred)
        cov_support.append(cov)
        if display == True:
            ax1.add_patch(patches.Circle(pred, 50, fill=True, lw=3))

    if display == True:
        fig.canvas.flush_events()
        ax1.set_xlim(0, 1000)
        ax1.set_ylim(0, 1000)
        plt.draw()
        ax1.cla()
    # breakpoint()

print("pred", pred_support[:5])
print("covariance", cov_support[:5])
print("real", real_support[:5])

plt.clf()
plt.cla()
plt.errorbar([float(x[0]) for x in pred_support], [float(x[1]) for x in pred_support], yerr=[float(x[0]) for x in cov_support], fmt='o-', label='Prediction', linewidth=7)
plt.plot([float(x[0]) for x in real_support], [float(x[1]) for x in real_support], label='Real', linewidth=1.0)
# plt.errorbar(x, y, yerr=y_err, fmt='o-', label='Measurement with Error Bars', capsize=3)
plt.legend()
plt.show()

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)

#
# if __name__ == '__main__':
#     # all train
#     args = parse_args()
#     display = args.display
#     phase = args.phase
#     total_time = 0.0
#     total_frames = 0
#     colours = np.random.rand(32, 3)  # used only for display
#     if (display):
#         if not os.path.exists('mot_benchmark'):
#             print(
#                 '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
#             exit()
#         plt.ion()
#         fig = plt.figure()
#         ax1 = fig.add_subplot(111, aspect='equal')
#
#     if not os.path.exists('output'):
#         os.makedirs('output')
#     pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
#     for seq_dets_fn in glob.glob(pattern):
#         mot_tracker = Sort(max_age=args.max_age,
#                            min_hits=args.min_hits,
#                            iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
#         seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
#         seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
#
#         with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:
#             print("Processing %s." % (seq))
#             for frame in range(int(seq_dets[:, 0].max())):
#                 frame += 1  # detection and frame numbers begin at 1
#                 dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
#                 dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
#                 total_frames += 1
#
#                 if (display):
#                     fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % (frame))
#                     im = io.imread(fn)
#                     ax1.imshow(im)
#                     plt.title(seq + ' Tracked Targets')
#
#                 start_time = time.time()
#                 trackers = mot_tracker.update(dets)
#                 cycle_time = time.time() - start_time
#                 total_time += cycle_time
#
#                 for d in trackers:
#                     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
#                           file=out_file)
#                     if (display):
#                         d = d.astype(np.int32)
#                         ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
#                                                         ec=colours[d[4] % 32, :]))
#
#                 if (display):
#                     fig.canvas.flush_events()
#                     plt.draw()
#                     ax1.cla()
#
#     print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
#     total_time, total_frames, total_frames / total_time))
#
#     if (display):
#         print("Note: to get real runtime results run without the option: --display")
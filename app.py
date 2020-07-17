import track
import copy
import pickle
import time
import sys
import zmq
import subprocess
from pycompss.api.parameter import *
from pycompss.api.task import task
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on, compss_open
from pycompss.api.binary import binary
from pycompss.api.constraint import constraint

try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle

colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255], [0, 0, 0], [255, 255, 255]]


@task(returns=list, trackers=INOUT)
def Track(listBoxes, dt, n_states, initial_age, age_threshold, trackers, referenceX, referenceY, identifier, nTrackers):
    newBoxes = []
    if identifier == 1:
        newBoxes = [t for t in listBoxes if t.x_pixel_ + t.w_ < referenceX and t.y_pixel_ + t.h_ < referenceY]
    elif identifier == 2:
        newBoxes = [t for t in listBoxes if t.x_pixel_ + t.w_ >= referenceX and t.y_pixel_ + t.h_ < referenceY]
    elif identifier == 3:
        newBoxes = [t for t in listBoxes if t.y_pixel_ + t.h_ >= referenceY]
    return track.Track2(newBoxes, dt, n_states, initial_age, age_threshold, trackers)


def createList(fn):
    listB = []
    for line in fn.splitlines():
        x, y, frame, classB, x_pixel, y_pixel, w, h = (line.decode('utf-8')).split(" ")
        listB.append(
            track.Data(float(x), float(y), int(frame), int(classB), float(x_pixel), float(y_pixel), float(w), float(h)))
    return listB


@constraint(AppSoftware="yolo")
@task(tracker1=IN, tracker2=IN, tracker3=IN)
def printFrame(frame, tracker1, tracker2, tracker3, time_task, counter):
    import cv2 as cv
    import os

    os.environ["DISPLAY"] = "localhost:10.0"
    cv.namedWindow('demo', cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty('demo', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    trackers = [tracker1, tracker2, tracker3]
    for i, ts in enumerate(trackers):
        for l in ts:
            x = int(l.traj_[-1].x_pixel_)
            y = int(l.traj_[-1].y_pixel_)
            w = int(l.traj_[-1].w_)
            h = int(l.traj_[-1].h_)
            textSize = cv.getTextSize(str(l.id_), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1);
            frame = cv.rectangle(frame, (x, y), ((x + textSize[0][0] - 2), (y - textSize[0][1] - 2)),
                                 (l.r_, l.g_, l.b_), -1);
            text = str(i) + " " + str(x) + ", " + str(y)
            frame = cv.putText(frame, str(l.id_), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (l.r_, l.g_, l.b_), 2)

    # addFPS
    counter += 1
    end_time = time.time()
    FPS = 0
    if (end_time - time_task > 0):
        FPS = counter / (time.time() - time_task)
        # counter = 0

    frame = cv.putText(frame, str(int(FPS)), (5, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    # printFrame
    cv.imshow('demo', frame)
    cv.waitKey(1)
    return end_time


def printCurrentTrackers(trackers):
    for i, ts in enumerate(trackers):
        print("Objects from TRACKER " + str(i))
        for l in ts:
            x = l.traj_[-1].x_
            y = l.traj_[-1].y_
            x_pixel = l.traj_[-1].x_pixel_
            y_pixel = l.traj_[-1].y_pixel_
            if len(l.pred_list_) > 0:
                vel_pred = l.pred_list_[-1].vel_
                yaw_pred = l.pred_list_[-1].yaw_
            else:
                vel_pred = "unknown"
                yaw_pred = "unknown"
            print("Object class: " + str(l.class_) + " Geolocation: (" + str(x) + ", " + str(y) + ") ID: " + str(
                l.id_) + " Pixel Position: " + str(x_pixel) + ", " + str(y_pixel) + " Speed predicted: " + str(
                vel_pred) + " Predicte Yaw: " + str(yaw_pred))


@task(frame=INOUT, listBoxes=INOUT, returns=list)
def frameProcessing(frame, listBoxes):
    context = zmq.Context()
    sink = context.socket(zmq.REQ)
    # sink.connect("tcp://192.168.50.103:5559")
    sink.connect("tcp://127.0.0.1:5559")
    sink.send(frame)
    fn = sink.recv()
    return createList(fn)


def executeTrackers(cap, referenceX, referenceY, video):
    trackers1 = []
    trackers2 = []
    trackers3 = []
    listBoxes = []

    initial_age = -5
    age_threshold = -8
    n_states = 5
    dt = 0.03

    i = 0
    counter = 0
    time_task = time.time()

    ret, frame = cap.read()

    # while(i < 100):
    while (1):
        listBoxes = frameProcessing(frame, listBoxes)

        trackers1 = Track(listBoxes, dt, n_states, initial_age, age_threshold, trackers1, referenceX, referenceY, 1, 3)
        trackers2 = Track(listBoxes, dt, n_states, initial_age, age_threshold, trackers2, referenceX, referenceY, 2, 3)
        trackers3 = Track(listBoxes, dt, n_states, initial_age, age_threshold, trackers3, referenceX, referenceY, 3, 3)

        old_frame = copy.deepcopy(frame)

        ret, frame = cap.read()
        if ret == False:
            cap = cv.VideoCapture(video)
            ret, frame = cap.read()
            continue

        time_task = printFrame(old_frame, trackers1, trackers2, trackers3, time_task, counter)

        # printCurrentTrackers([trackers1, trackers2, trackers3])
        if i % 5 == 0:
            compss_barrier()
        i += 1


if __name__ == "__main__":
    import cv2 as cv

    if len(sys.argv) == 2:
        video = sys.argv[1]
    else:
        video = "scenario1_video_1.mp4"

    cap = cv.VideoCapture(video)

    referenceX = cap.get(cv.CAP_PROP_FRAME_WIDTH) / 2.0
    referenceY = cap.get(cv.CAP_PROP_FRAME_HEIGHT) / 2.0

    executeTrackers(cap, referenceX, referenceY, video)

    cap.release()
    print("Exiting Application...")
    sys.exit(0)
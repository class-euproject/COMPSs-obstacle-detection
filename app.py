from pycompss.api.parameter import *
from pycompss.api.task import task
from pycompss.api.api import compss_barrier, compss_wait_on
from collections import deque
from typing import Tuple, List
import track
from datetime import datetime

QUAD_REG_LEN = 20
QUAD_REG_OFFSET = 5


# @constraint(AppSoftware="yolo")
@task(returns=list, listBoxes=IN, trackers=IN, tracker_indexes=IN, cur_index=IN)
def execute_tracking(list_boxes, trackers, tracker_indexes, cur_index):
    import pymap3d as pm
    trackers, tracker_indexes, cur_index = track.track2(list_boxes, trackers, tracker_indexes, cur_index)
    for tracker in trackers:
        obj = tracker.traj[-1]
        obj.x, obj.y, _ = pm.enu2geodetic(obj.x, obj.y, 0, 44.655540, 10.934315, 0)
    return trackers, tracker_indexes, cur_index


# @constraint(AppSoftware="yolo")
@task(returns=Tuple[bool, list],)
def receive_boxes():
    import zmq
    import struct

    context = zmq.Context()
    sink = context.socket(zmq.REP)
    sink.connect("tcp://127.0.0.1:5559")  # tcp://172.0.0.1 for containerized executions

    double_size = 8
    int_size = float_size = 4

    boxes = []
    message = sink.recv()
    sink.send_string("")
    flag = len(message) > 0

    for offset in range(1, len(message), double_size * 2 + int_size + 1 + float_size * 4):
        coord_north, coord_east, frame_number, obj_class = struct.unpack_from('ddIc', message[
                                                                                      offset:offset + double_size * 2 + int_size + 1])
        x, y, w, h = struct.unpack_from('ffff', message[offset + double_size * 2 + int_size + 1:])
        # print((coord_north, coord_east, frame_number, ord(obj_class), x, y, w, h))
        boxes.append(track.obj_m(x, y, frame_number, ord(obj_class), int(w), int(h)))
    return flag, boxes


def print_trackers(tracker1, tracker2, tracker3):
    trackers = [tracker1, tracker2, tracker3]
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

            print("Object class: " + str(l.class_) + " Geolocation: (" + str(x) + ", " + str(y) + ") ID: " + str(l.id_)
                  + " Pixel Position: " + str(x_pixel) + ", " + str(y_pixel) + " Speed predicted: " + str(vel_pred) +
                  " Predicted Yaw: " + str(yaw_pred))


@task(returns=int, trackers=IN, count=IN)
def federate_info(trackers, count):
    import uuid
    import os
    os.environ["DATACLAYSESSIONCONFIG"] = "/tmp/pycharm_project_225/cfgfiles/session.properties"
    os.environ["DATACLAYCLIENTCONFIG"] = "/tmp/pycharm_project_225/cfgfiles/client.properties"
    from dataclay.api import init, finish, register_dataclay
    from dataclay.exceptions.exceptions import DataClayException
    # init("/tmp/pycharm_project_225/cfgfiles/session.properties")
    init()

    from CityNS.classes import Event, Object, EventsSnapshot
    classes = ["person", "car", "truck", "bus", "motor", "bike", "rider", "traffic light", "traffic sign", "train"]
    snapshot_alias = "events_" + str(count)
    snapshot = EventsSnapshot(snapshot_alias)
    snapshot.make_persistent(alias=snapshot_alias)

    dataclay_cloud = register_dataclay("192.168.7.32", 11034)
    for tracker in trackers:
        vel_pred = tracker.predList[-1].vel if len(tracker.predList) > 0 else -1.0
        yaw_pred = tracker.predList[-1].yaw if len(tracker.predList) > 0 else -1.0
        lat = tracker.traj[-1].x
        lon = tracker.traj[-1].y

        event = Event(uuid.uuid4().int, int(datetime.now().timestamp() * 1000), lon, lat)
        print(f"Registering object alias {tracker.id}")
        object_alias = "obj_" + str(tracker.id)
        try:
            event_object = Object.get_by_alias(object_alias)
        except DataClayException as e:
            event_object = Object(tracker.id, classes[tracker.cl], vel_pred, yaw_pred)
            event_object.make_persistent(alias=object_alias)

        event_object.add_event(event)
        event_object.federate(dataclay_cloud)
        snapshot.add_object_refs(object_alias)

    try:
        snapshot.federate(dataclay_cloud)
        pass
    except DataClayException as e:
        print(e)
    # finish()
    return count


def execute_trackers():
    tracker1 = []
    tracker2 = []
    tracker3 = []

    tracker_indexes = []
    cur_index = 0

    video_resolution = (1920, 1080)  # TODO: RESOLUTION SHOULD NOT BE HARDCODED!
    reference_x, reference_y = [r // 2 for r in video_resolution]

    i = 0
    dummy = 0
    ret = True
    while ret:
        ret, list_boxes = compss_wait_on(receive_boxes())

        if ret:
            tracker1, tracker_indexes, cur_index = compss_wait_on(execute_tracking([t for t in list_boxes if t.x + t.w < reference_x and t.y + t.h < reference_y], tracker1, tracker_indexes, cur_index))
            tracker2, tracker_indexes, cur_index = compss_wait_on(execute_tracking([t for t in list_boxes if t.x + t.w >= reference_x and t.y + t.h < reference_y], tracker2, tracker_indexes, cur_index))
            tracker3, tracker_indexes, cur_index = compss_wait_on(execute_tracking([t for t in list_boxes if t.y + t.h >= reference_y], tracker3, tracker_indexes, cur_index))

            dummy = federate_info(tracker1 + tracker2 + tracker3, i)
            i += 1
            if i % 5 == 0:
                compss_barrier()


def main():
    import sys
    import time

    start_time = time.time()
    execute_trackers()
    print("ExecTime: " + str(time.time() - start_time))

    print("Exiting Application...")
    sys.exit(0)


if __name__ == "__main__":
    main()

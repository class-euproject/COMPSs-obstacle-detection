from pycompss.api.parameter import *
from pycompss.api.task import task
from pycompss.api.api import compss_barrier, compss_wait_on
from collections import deque
from typing import Tuple, List
import track
from datetime import datetime

QUAD_REG_LEN = 20
QUAD_REG_OFFSET = 5


def pixel2GPS(x, y):
    import pymap3d as pm
    lat, lon, _ = pm.enu2geodetic(x, y, 0, 44.655540, 10.934315, 0)
    return lat, lon


# @constraint(AppSoftware="yolo")
@task(returns=Tuple, list_boxes=IN, filter_fn=IN, trackers=IN, tracker_indexes=IN, cur_index=IN)
def execute_tracking(list_boxes, filter_fn, trackers, tracker_indexes, cur_index):
    return track.track2([t for t in list_boxes if filter_fn(t)], trackers, tracker_indexes, cur_index)


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
    return boxes


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


def trigger_openwhisk(alias):
    import requests
    APIHOST = 'https://192.168.7.40:31001'
    AUTH_KEY = '23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP'
    NAMESPACE = '_'
    BLOCKING = 'true'
    RESULT = 'true'
    TRIGGER = 'tp-trigger'
    url = APIHOST + '/api/v1/namespaces/' + NAMESPACE + '/triggers/' + TRIGGER
    user_pass = AUTH_KEY.split(':')
    requests.post(url, params={'blocking': BLOCKING, 'result': RESULT}, json={"ALIAS": str(alias)},
                             auth=(user_pass[0], user_pass[1]), verify=False)


@task(returns=int, tracker1=COLLECTION_IN, tracker2=COLLECTION_IN, tracker3=COLLECTION_IN, count=IN, dummy=IN)
def federate_info(tracker1, tracker2, tracker3, count, dummy):
    import uuid
    import os
    # os.environ["DATACLAYSESSIONCONFIG"] = "/opt/cfgfiles/session.properties"
    # os.environ["DATACLAYCLIENTCONFIG"] = "/opt/cfgfiles/client.properties"
    os.environ["DATACLAYSESSIONCONFIG"] = "/tmp/pycharm_project_225/cfgfiles/session.properties"
    os.environ["DATACLAYCLIENTCONFIG"] = "/tmp/pycharm_project_225/cfgfiles/client.properties"
    from dataclay.api import init, finish, register_dataclay
    from dataclay.exceptions.exceptions import DataClayException
    # init("/tmp/pycharm_project_225/cfgfiles/session.properties")
    init()

    from CityNS.classes import Event, Object, EventsSnapshot, DKB
    kb = DKB.get_by_alias("DKB")

    classes = ["person", "car", "truck", "bus", "motor", "bike", "rider", "traffic light", "traffic sign", "train"]
    snapshot_alias = "events_" + str(count)
    snapshot = EventsSnapshot(snapshot_alias)
    print(f"Persisting {snapshot_alias}")
    snapshot.make_persistent(alias=snapshot_alias)

    # dataclay_cloud = register_dataclay("192.168.7.32", 11034)
    trackers = tracker1 + tracker2 + tracker3
    for index, tracker in enumerate(trackers):
        vel_pred = tracker.predList[-1].vel if len(tracker.predList) > 0 else -1.0
        yaw_pred = tracker.predList[-1].yaw if len(tracker.predList) > 0 else -1.0
        lat, lon = pixel2GPS(tracker.traj[-1].x, tracker.traj[-1].y)
        # lat = tracker.traj[-1].x
        # lon = tracker.traj[-1].y

        event = Event(uuid.uuid4().int, int(datetime.now().timestamp() * 1000), float(lon), float(lat))
        print(f"Registering object alias {tracker.id}")
        object_alias = "obj_" + str(index)
        try:
            event_object = Object.get_by_alias(object_alias)
        except DataClayException as e:
            event_object = Object(tracker.id, classes[tracker.cl], vel_pred, yaw_pred)
            print(f"Persisting {object_alias}")
            event_object.make_persistent(alias=object_alias)

        event_object.add_event(event)
        # event_object.federate(dataclay_cloud)
        snapshot.add_object_refs(object_alias)

    kb.add_events_snapshot(snapshot)
    trigger_openwhisk(snapshot_alias)

    """
    try:
        snapshot.federate(dataclay_cloud)
    except DataClayException as e:
        print(e)
    """
    # finish()
    return dummy


def execute_trackers():
    tracker1 = []
    tracker2 = []
    tracker3 = []

    tracker_indexes1 = []
    tracker_indexes2 = []
    tracker_indexes3 = []
    cur_index1 = 0
    cur_index2 = 0
    cur_index3 = 0

    # video_resolution = (1920, 1080)  # TODO: RESOLUTION SHOULD NOT BE HARDCODED!
    video_resolution = (3072, 1730)  # TODO: RESOLUTION SHOULD NOT BE HARDCODED!
    reference_x, reference_y = [r // 2 for r in video_resolution]

    i = 0
    dummy = 0
    while True:
        list_boxes = receive_boxes()

        trackers = tracker1 + tracker2 + tracker3
        # tracker1, tracker_indexes1, cur_index1 = compss_wait_on(execute_tracking(list_boxes, lambda t: t.x + t.w < reference_x and t.y + t.h < reference_y, trackers, tracker_indexes1, cur_index1))
        results1 = execute_tracking(list_boxes, lambda t: t.x + t.w < reference_x and t.y + t.h < reference_y, tracker1, tracker_indexes1, cur_index1)
        results2 = execute_tracking(list_boxes, lambda t: t.x + t.w >= reference_x and t.y + t.h < reference_y, tracker2, tracker_indexes2, cur_index2)
        results3 = execute_tracking(list_boxes, lambda t: t.y + t.h >= reference_y, tracker3, tracker_indexes3, cur_index3)

        tracker1, tracker_indexes1, cur_index1 = compss_wait_on(results1)
        tracker2, tracker_indexes2, cur_index2 = compss_wait_on(results2)
        tracker3, tracker_indexes3, cur_index3 = compss_wait_on(results3)
        dummy = federate_info(tracker1, tracker2, tracker3, i, compss_wait_on(dummy))
        # The dummy variable enforces dependency between federate_info tasks

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

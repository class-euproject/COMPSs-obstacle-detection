from pycompss.api.parameter import *
from pycompss.api.task import task
from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.constraint import constraint
from utils import pixel2GPS
import track
import deduplicator as dd
from datetime import datetime

# import threading

NUM_ITERS = 100
N = 5


# @constraint(AppSoftware="nvidia")
@task(returns=3, list_boxes=IN, trackers=IN, tracker_indexes=IN, cur_index=IN)
def execute_tracking(list_boxes, trackers, tracker_indexes, cur_index):
    return track.track2(list_boxes, trackers, tracker_indexes, cur_index)


"""
# @constraint(AppSoftware="nvidia")
@task(returns=4,)
def receive_boxes(socket_ip, dummy):
    import zmq
    import struct

    if ":" not in socket_ip:
        socket_ip += ":5559"

    context = zmq.Context()
    sink = context.socket(zmq.REP)

    # sink.connect("tcp://127.0.0.1:5559")
    sink.connect(f"tcp://{socket_ip}")  # tcp://172.0.0.1 for containerized executions

    double_size = unsigned_long_size = 8
    int_size = float_size = 4

    boxes = []
    message = sink.recv()
    sink.send_string("")
    flag = len(message) > 0
    # This flag serves to know if the video has ended
    cam_id = struct.unpack_from("i", message[1:1 + int_size])[0]
    timestamp = struct.unpack_from("Q", message[1 + int_size:1 + int_size + unsigned_long_size])[0]

    for offset in range(1 + int_size + unsigned_long_size, len(message), double_size * 2 + int_size + 1 + float_size * 4):
        north, east, frame_number, obj_class = struct.unpack_from('ddIc', message[
                                                                          offset:offset + double_size * 2 + int_size + 1])
        x, y, w, h = struct.unpack_from('ffff', message[offset + double_size * 2 + int_size + 1:])
        # print((coord_north, coord_east, frame_number, ord(obj_class), x, y, w, h))
        boxes.append(track.obj_m(north, east, frame_number, ord(obj_class), int(w), int(h)))
    return cam_id, timestamp, boxes, dummy
"""


# @constraint(AppSoftware="nvidia")
@task(returns=4, )
def receive_boxes(pipe_paths, dummy):
    import struct

    double_size = unsigned_long_size = 8
    int_size = float_size = 4

    # opening and closing pipes at each task otherwise read gets blocked until no writer
    fifo_read = open(pipe_paths[0], 'rb')
    fifo_write = open(pipe_paths[1], 'w')

    boxes = []
    message = fifo_read.read()
    fifo_read.close()
    fifo_write.write('0')
    fifo_write.close()
    flag = (message == 0)
    # This flag serves to know if the video has ended
    cam_id = struct.unpack_from("i", message[1:1 + int_size])[0]
    timestamp = struct.unpack_from("Q", message[1 + int_size:1 + int_size + unsigned_long_size])[0]
    for offset in range(1 + int_size + unsigned_long_size, len(message),
                        double_size * 2 + int_size + 1 + float_size * 4):
        north, east, frame_number, obj_class = struct.unpack_from('ddIc', message[
                                                                          offset:offset + double_size * 2 + int_size + 1])
        x, y, w, h = struct.unpack_from('ffff', message[offset + double_size * 2 + int_size + 1:])
        boxes.append(track.obj_m(north, east, frame_number, ord(obj_class), int(w), int(h)))
    return cam_id, timestamp, boxes, dummy


@task(returns=3, )
def merge_tracker_state(trackers_list, cur_index):
    import itertools
    trackers = []
    prev_cur_index = cur_index
    for tracker in itertools.chain.from_iterable(trackers_list):
        if tracker.id >= prev_cur_index:
            tracker.id = cur_index
            cur_index += 1
        trackers.append(tracker)
    tracker_indexes = [True] * cur_index + [False] * (32767 - cur_index)
    return trackers, tracker_indexes, cur_index


# @constraint(AppSoftware="nvidia")
@task(trackers_list=COLLECTION_IN)
# def deduplicate(trackers_list, cam_ids, timestamps): # TODO: waiting for UNIMORE
def deduplicate(trackers_list):
    return_message = dd.compute_deduplicator(trackers_list)
    # return_message = dd.compute_deduplicator(trackers_list, cam_ids, timestamps) # TODO: waiting for UNIMORE
    # print(f"Returned {len(return_message)} objects (from the original "
    #      f"{' + '.join([str(len(t)) for t in trackers_list])} = {sum([len(t) for t in trackers_list])})")
    return return_message


def dump(id_cam, ts, trackers, iteration):
    import pygeohash as pgh
    import os
    if not os.path.exists("singlecamera.in"):
        f = open("singlecamera.in", "w+")
        f.close()
    with open("singlecamera.in", "a+") as f:
        for tracker in trackers:
            lat, lon = pixel2GPS(tracker.traj[-1].x, tracker.traj[-1].y)
            geohash = pgh.encode(lat, lon, precision=7)
            speed = abs(tracker.predList[-1].vel) if len(tracker.predList) > 0 else 0.0
            speed = speed / 2
            yaw = tracker.predList[-1].yaw if len(tracker.predList) > 0 else 0.0
            f.write(
                f"{id_cam} {iteration} {ts} {tracker.cl} {lat} {lon} {geohash} {speed} {yaw} {id_cam}_{tracker.id}\n")


@task(returns=1, trackers_list=COLLECTION_IN, count=IN, kb=IN, list_objects=IN)
def persist_info_accumulated(trackers_list, count, kb, list_objects):
    from CityNS.classes import EventsSnapshot
    snapshot_alias = "events_" + str(count)
    snapshot = EventsSnapshot(snapshot_alias)
    kb.add_events_snapshot(snapshot)  # persists snapshot
    for trackers in trackers_list:
        snapshot.add_events_from_trackers(trackers, list_objects)  # create events inside dataclay
    return snapshot


@constraint(AppSoftware="nvidia")
@task(returns=1, trackers=IN, count=IN, kb=IN, list_objects=IN)
def persist_info(trackers, count, kb, list_objects):
    from CityNS.classes import EventsSnapshot
    snapshot_alias = "events_" + str(count)
    snapshot = EventsSnapshot(snapshot_alias)
    kb.add_events_snapshot(snapshot)  # persists snapshot
    # print("LEN OF TRACKERS: " + str(len(trackers[1])))
    snapshot.add_events_from_trackers(trackers, list_objects)  # create events inside dataclay
    return snapshot


@task(snapshot=IN)
def federate_info(snapshot):
    from dataclay.api import get_dataclay_id
    from dataclay.exceptions.exceptions import DataClayException
    from CityNS.classes import Object

    print(f"Starting federation of snapshot {snapshot.snap_alias}")
    federation_ip, federation_port = "192.168.7.32", 21034  # TODO: change port accordingly
    dataclay_to_federate = get_dataclay_id(federation_ip, federation_port)
    for obj_alias in snapshot.objects_refs:
        obj = Object.get_by_alias(obj_alias)
        if len(obj.get_federation_targets() or []) == 0:
            try:
                obj.federate(dataclay_to_federate)
                print(f"Federation of object {obj_alias} was successful")
            except KeyboardInterrupt as e:
                raise e
            except DataClayException as e:
                print(f"Federation of object {obj_alias} failed")
                print(e.args[0])
        try:
            obj.events_history[-1].federate(dataclay_to_federate)
            print(f"Federation of last event of object {obj_alias} was successful")
        except KeyboardInterrupt as e:
            raise e
        except DataClayException as e:
            print(f"Federation of last event of object {obj_alias} failed")
            print(e.args[0])
    snapshot.federate(dataclay_to_federate)
    print("Finished federation")


# @constraint(AppSoftware="phemlight") # TODO: to be executed in Cloud. Remove it otherwise
@task(input_path=IN, output_file=IN)
def analyze_pollution(input_path, output_file):
    import os
    import uuid
    pollution_file_name = "pollution_" + str(uuid.uuid4()).split("-")[-1] + ".csv"
    if os.path.exists(pollution_file_name):
        os.remove(pollution_file_name)
    from CityNS.classes import Event, Object, EventsSnapshot, DKB
    kb = DKB.get_by_alias("DKB")
    obj_refs = set()
    i = 0
    with open(pollution_file_name, "w") as f:
        f.write("VehID, LinkID, Time, Vehicle_type, Av_link_speed\n")
        for snap in kb.kb:
            for obj_ref in snap.objects_refs:
                if obj_ref not in obj_refs:
                    obj_refs.add(obj_ref)
                    obj = Object.get_by_alias(obj_ref)
                    obj_type = obj.type
                    if obj_type in ["car", "bus"]:
                        obj_type = obj_type.title()
                    elif obj_type == "truck":
                        obj_type = "HDV"
                    else:
                        continue
                    for event in obj.events_history:
                        f.write(f"{obj_ref}, {20939 + i % 2}, {event.timestamp}, {obj_type}, 50\n")  # TODO: link_id
                        # needs to be obtained from object
                        i += 1
    os.system(
        f"Rscript --vanilla /home/nvidia/CLASS/class-app/phemlight/PHEMLight_advance.R {input_path} $PWD/{pollution_file_name}"
        f" {output_file}")  # TODO: R script path is hardcoded


@task(is_replicated=True)
def init_task():
    import uuid
    from CityNS.classes import DKB, Event, Object, EventsSnapshot, ListOfEvents, ListOfObjects
    kb = DKB()
    kb.make_persistent("FAKE_" + str(uuid.uuid4()))
    kb.get_objects_from_dkb()
    snap = EventsSnapshot("FAKE_SNAP_" + str(uuid.uuid4()))
    snap.make_persistent("FAKE_SNAP_" + str(uuid.uuid4()))
    snap.get_objects_refs()
    event = Event(None, None, None, None, None, None, None)
    event.make_persistent("FAKE_EVENT_" + str(uuid.uuid4()))
    obj = Object("FAKE_OBJ_" + str(uuid.uuid4()), "FAKE")
    obj.make_persistent("FAKE_OBJ_" + str(uuid.uuid4()))
    obj.get_events_history()
    list_objs = ListOfObjects()
    list_objs.make_persistent("FAKE_LISTOBJ_" + str(uuid.uuid4()))
    list_objs.initialize_locks()
    list_events = ListOfEvents([])
    list_events.make_persistent("FAKE_LISTEV_" + str(uuid.uuid4()))


@constraint(AppSoftware="nvidia")
@task(returns=3, trackers_list=IN, tracker_indexes=IN, cur_index=IN)
def boxes_and_track(socket_ip, trackers_list, tracker_indexes, cur_index):
    _, _, list_boxes, _ = receive_boxes(socket_ip, 0)
    return execute_tracking(list_boxes, trackers_list, tracker_indexes, cur_index)


def execute_trackers(socket_ips, kb, list_objects):
    import uuid
    import time
    import sys
    import os

    trackers_list = [[]] * len(socket_ips)
    tracker_indexes = [[]] * len(socket_ips)
    cur_index = [0] * len(socket_ips)
    cam_ids = [0] * len(socket_ips)
    timestamps = [0] * len(socket_ips)

    # video_resolution = (1920, 1080)  # TODO: RESOLUTION SHOULD NOT BE HARDCODED!
    video_resolution = (3072, 1730)  # TODO: RESOLUTION SHOULD NOT BE HARDCODED!
    reference_x, reference_y = [r // 2 for r in video_resolution]

    """
    if os.path.exists("/tmp/pipe_yolo2COMPSs"):
            os.unlink("/tmp/pipe_yolo2COMPSs")
    if os.path.exists("/tmp/pipe_COMPSs2yolo"):
            os.unlink("/tmp/pipe_COMPSs2yolo")
    try:
        os.mkfifo("/tmp/pipe_yolo2COMPSs")
    except OSError as e:
        pass
    try:
        os.mkfifo("/tmp/pipe_COMPSs2yolo")
    except OSError as e:
        pass

    fifo_read = open("/tmp/pipe_yolo2COMPSs", 'rb')
    fifo_write = open("/tmp/pipe_COMPSs2yolo", 'w')

    # create pipes from COMPSs to yolo
    pipe_paths = ["/tmp/pipe_COMPSs2yolo"]
    for pipe_path in pipe_paths:
        if os.path.exists(pipe_path):
            os.unlink(pipe_path)
        try:
            os.mkfifo(pipe_path)
        except OSError as e:
            print("Failed to create FIFO: " + str(e))
            pass
    """

    pipe_paths = [("/tmp/pipe_yolo2COMPSs", "/tmp/pipe_COMPSs2yolo")] # TODO: define in main and pass it instead of socket
    start_time = time.time()

    deduplicated_trackers_list = []  # TODO: accumulate trackers
    i = 0
    reception_dummies = [0] * len(socket_ips)
    while i < NUM_ITERS:
        for index, pipe_path in enumerate(pipe_paths):
            cam_ids[index], timestamps[index], list_boxes, reception_dummies[index] = \
                receive_boxes(pipe_path, reception_dummies[index])
            trackers_list[index], tracker_indexes[index], cur_index[index] = execute_tracking(list_boxes,
                                                                                              trackers_list[index],
                                                                                              tracker_indexes[index],
                                                                                              cur_index[index])
            # print(f"CAM ID: {cam_ids[index]}, timestamp: {timestamps[index]}, list_boxes: {[list_boxes]}")
        """
        for index, socket_ip in enumerate(socket_ips):
            # trackers_list[index], tracker_indexes[index], cur_index[index] = \
            #        boxes_and_track(socket_ip, trackers_list[index], tracker_indexes[index], cur_index[index])
            cam_ids[index], timestamps[index], list_boxes, reception_dummies[index] = \
                receive_boxes(socket_ip, reception_dummies[index])
            trackers_list[index], tracker_indexes[index], cur_index[index] = \
                execute_tracking(list_boxes, trackers_list[index], tracker_indexes[index], cur_index[index])
            # dump(cam_ids[index], timestamps[index], trackers_list[index], i)
        # print([(t.id, t.predList[-1].vel / (3.6 * 2)) for t in trackers_list[0] if len(t.predList) > 0
        # and abs(t.predList[-1].vel) > .1])
        """

        # trackers, tracker_indexes, cur_index = merge_tracker_state(trackers_list)
        deduplicated_trackers = deduplicate(trackers_list) # , cam_ids, timestamps) # TODO: pass cam_ids and timestamps
        # deduplicated_trackers_list.append(deduplicated_trackers) # TODO: accumulate trackers

        """# TODO: accumulate trackers
        if i != 0 and (i+1) % N == 0:
            snapshot = persist_info_accumulated(deduplicated_trackers_list, i, kb, list_objects)
            deduplicated_trackers_list.clear() 
        """
        snapshot = persist_info(deduplicated_trackers, i, kb, list_objects)
        i += 1
        # identify from which camera is each object detected and tracked
        federate_info(snapshot)
        if i != 0 and i % 10 == 0:
            input_path = "/home/nvidia/CLASS/class-app/phemlight/in/"
            output_file = "results_" + str(uuid.uuid4()).split("-")[-1] + ".csv"
            analyze_pollution(input_path, output_file)

    compss_barrier()
    end_time = time.time()
    print("Exec Inner Time: " + str(end_time - start_time))
    print("Exec Inner Time per Iteration: " + str((end_time - start_time) / NUM_ITERS))

def main():
    import time
    from dataclay.api import init, register_dataclay, finish
    from dataclay.exceptions.exceptions import DataClayException

    init()
    from CityNS.classes import DKB, ListOfObjects
    register_dataclay("192.168.7.32", 21034)

    # initialize all computing units in all workers
    num_cus = 8
    for i in range(num_cus):
        init_task()
    compss_barrier()

    list_objects = ListOfObjects()
    list_objects.make_persistent()
    list_objects.initialize_locks()

    try:
        kb = DKB.get_by_alias("DKB")
    except DataClayException:
        kb = DKB()
        kb.make_persistent("DKB")

    start_time = time.time()
    # execute_trackers(["192.168.50.103", "192.168.50.103:5558", "192.168.50.103:5557"], kb, list_objects)
    # execute_trackers(["192.168.50.103", "192.168.50.103:5558"], kb, list_objects)
    execute_trackers(["192.168.50.103"], kb, list_objects)
    # compss_barrier()

    # print("ExecTime: " + str(time.time() - start_time))
    # print("ExecTime per Iteration: " + str((time.time() - start_time) / NUM_ITERS))

    print("Exiting Application...")
    finish()


if __name__ == "__main__":
    main()

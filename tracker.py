from pycompss.api.parameter import *
from pycompss.api.task import task
from pycompss.api.api import compss_barrier, compss_wait_on
from utils import pixel2GPS
import track
import deduplicator as dd
from datetime import datetime


# @constraint(AppSoftware="yolo")
@task(returns=3, list_boxes=IN, trackers=IN, tracker_indexes=IN, cur_index=IN)
def execute_tracking(list_boxes, trackers, tracker_indexes, cur_index):
    return track.track2(list_boxes, trackers, tracker_indexes, cur_index)


# @constraint(AppSoftware="yolo")
@task(returns=3,)
def receive_boxes(socket_ip, dummy):
    import zmq
    import pymap3d as pm
    import struct

    if ":" not in socket_ip:
        socket_ip += ":5559"

    context = zmq.Context()
    sink = context.socket(zmq.REP)

    # sink.connect("tcp://127.0.0.1:5559")
    sink.connect(f"tcp://{socket_ip}")  # tcp://172.0.0.1 for containerized executions

    double_size = 8
    int_size = float_size = 4

    boxes = []
    message = sink.recv()
    sink.send_string("")
    flag = len(message) > 0
    # This flag serves to know if the video has ended
    cam_id = struct.unpack_from("i", message[1:1 + int_size])[0]

    for offset in range(1 + int_size, len(message), double_size * 2 + int_size + 1 + float_size * 4):
        north, east, frame_number, obj_class = struct.unpack_from('ddIc', message[
                                                                                      offset:offset + double_size * 2 + int_size + 1])
        x, y, w, h = struct.unpack_from('ffff', message[offset + double_size * 2 + int_size + 1:])
        # print((coord_north, coord_east, frame_number, ord(obj_class), x, y, w, h))
        boxes.append(track.obj_m(north, east, frame_number, ord(obj_class), int(w), int(h)))
    return cam_id, boxes, dummy


@task(returns=3,)
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


@task(trackers_list=COLLECTION_IN)
def deduplicate(trackers_list):
    return_message = dd.compute_deduplicator(trackers_list)
    print(f"Returned {len(return_message)} objects (from the original "
          f"{' + '.join([str(len(t)) for t in trackers_list])} = {sum([len(t) for t in trackers_list])})")
    return return_message


def dump(id_cam, trackers, iteration):
    import pygeohash as pgh
    import os
    ts = int(datetime.now().timestamp() * 1000)
    if not os.path.exists("singlecamera.in"):
        f = open("singlecamera.in", "w+")
        f.close()
    with open("singlecamera.in", "a+") as f:
        for tracker in trackers:
            lat, lon = pixel2GPS(tracker.traj[-1].x, tracker.traj[-1].y)
            geohash = pgh.encode(lat, lon, precision=7)
            speed = abs(tracker.predList[-1].vel) if len(tracker.predList) > 0 else -1.0
            speed = speed / 2 if speed != -1.0 else -1.0
            yaw = tracker.predList[-1].yaw if len(tracker.predList) > 0 else -1.0
            f.write(f"{id_cam} {iteration} {ts} {tracker.cl} {lat} {lon} {geohash} {speed} {yaw} {id_cam}_{tracker.id}\n")


@task(returns=2, trackers=IN, count=IN, dummy=IN)
#def persist_info(trackers, count, dummy, id_cam): # TODO: id_cam should be passed in order to identify which objects 
#                                                  have been identified by each cam for the alias: {id_cam}_{tracker.id}
def persist_info(trackers, count, dummy):
    import uuid
    import pygeohash as pgh
    from dataclay.exceptions.exceptions import DataClayException

    from CityNS.classes import Event, Object, EventsSnapshot, DKB
    kb = DKB.get_by_alias("DKB")

    classes = ["person", "car", "truck", "bus", "motor", "bike", "rider", "traffic light", "traffic sign", "train"]
    snapshot_alias = "events_" + str(count)
    snapshot = EventsSnapshot(snapshot_alias)
    print(f"Persisting {snapshot_alias}")
    snapshot.make_persistent(alias=snapshot_alias)
    snapshot_ts = int(datetime.now().timestamp() * 1000)
    objects = []

    for index, tracker in enumerate(trackers):
        vel_pred = tracker.predList[-1].vel if len(tracker.predList) > 0 else -1.0
        yaw_pred = tracker.predList[-1].yaw if len(tracker.predList) > 0 else -1.0
        lat, lon = pixel2GPS(tracker.traj[-1].x, tracker.traj[-1].y)

        print(f"Registering object alias {tracker.id}")
        object_alias = "obj_" + str(index) # TODO: check if correct
        #object_alias = str(id_cam) + "_" + str(tracker.id) # TODO: check if possible and replace line above
        try:
            event_object = Object.get_by_alias(object_alias)
        except DataClayException:
            event_object = Object(object_alias, classes[tracker.cl])
            print(f"Persisting {object_alias}")
            event_object.make_persistent(alias=object_alias)

        event_object.geohash = pgh.encode(lat, lon)
        event = Event(uuid.uuid4().int, event_object, snapshot_ts, vel_pred, yaw_pred, float(lon), float(lat))
        event_object.add_event(event)
        snapshot.add_object_refs(object_alias)
        objects.append(event_object)

    kb.add_events_snapshot(snapshot)
    # trigger_openwhisk(snapshot_alias)
    return dummy, snapshot


@task(snapshot=IN)
def federate_info(snapshot):
    from dataclay.api import get_dataclay_id
    from dataclay.exceptions.exceptions import DataClayException
    from CityNS.classes import Object

    print(f"Starting federation of snapshot {snapshot.snap_alias}")
    federation_ip, federation_port = "192.168.7.32", 21034 # TODO: change port accordingly
    dataclay_to_federate = get_dataclay_id(federation_ip, federation_port)
    for obj_alias in snapshot.objects_refs:
        object = Object.get_by_alias(obj_alias)
        if len(object.get_federation_targets() or []) == 0:
            try:
                object.federate(dataclay_to_federate)
                print(f"Federation of object {obj_alias} was successful")
            except KeyboardInterrupt as e:
                raise e
            except DataClayException as e:
                print(f"Federation of object {obj_alias} failed")
                print(e.args[0])
        try:
            object.events_history[-1].federate(dataclay_to_federate)
            print(f"Federation of last event of object {obj_alias} was successful")
        except KeyboardInterrupt as e:
            raise e
        except DataClayException as e:
            print(f"Federation of last event of object {obj_alias} failed")
            print(e.args[0])
    snapshot.federate(dataclay_to_federate)
    print("Finished federation")


@task(input_path=IN, output_file=FILE_OUT)
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
                        f.write(f"{obj_ref}, {20939 + i % 2}, {event.timestamp}, {obj_type}, 50\n")
                        i += 1
    os.system(f"Rscript --vanilla /home/nvidia/CLASS/class-app/phemlight/PHEMLight_advance.R {input_path} $PWD/{pollution_file_name}"
              f" {output_file}")  # TODO: R script path is hardcoded


def execute_trackers(socket_ips):
    import uuid
    trackers_list = [[]] * len(socket_ips)

    tracker_indexes = [[]] * len(socket_ips)
    cur_index = [0] * len(socket_ips)

    # video_resolution = (1920, 1080)  # TODO: RESOLUTION SHOULD NOT BE HARDCODED!
    video_resolution = (3072, 1730)  # TODO: RESOLUTION SHOULD NOT BE HARDCODED!
    reference_x, reference_y = [r // 2 for r in video_resolution]

    i = 0
    reception_dummies = [0] * len(socket_ips)
    persist_dummy = 0
    # while 1:
    while i < 100:
        for index, socket_ip in enumerate(socket_ips):
            cam_id, list_boxes, reception_dummies[index] = receive_boxes(socket_ip, reception_dummies[index])
            trackers_list[index], tracker_indexes[index], cur_index[index] = execute_tracking(list_boxes, trackers_list[index], tracker_indexes[index], cur_index[index])
            # dump(cam_id, trackers_list[index], i)
        # print([(t.id, t.predList[-1].vel / (3.6 * 2)) for t in trackers_list[0] if len(t.predList) > 0 and abs(t.predList[-1].vel) > .1])

        # trackers, tracker_indexes, cur_index = merge_tracker_state(trackers_list)
        deduplicated_trackers = deduplicate(trackers_list)

        persist_dummy, snapshot = persist_info(deduplicated_trackers, i, persist_dummy) # TODO: we need a way to identify from which camera is each object detected and tracked
        federate_info(snapshot)
        # The dummy variable enforces dependency between federate_info tasks

        i += 1
        if i % 10 == 0:
            compss_barrier() #TODO: uncomment below
            """
            input_path = "/home/nvidia/CLASS/class-app/phemlight/in/"
            output_file = "results_" + str(uuid.uuid4()).split("-")[-1] + ".csv"
            analyze_pollution(input_path, output_file)
            """


def main():
    import time
    from dataclay.api import init, register_dataclay, finish
    from dataclay.exceptions.exceptions import DataClayException

    init()
    from CityNS.classes import DKB
    register_dataclay("192.168.7.32", 21034)
    try:
        DKB.get_by_alias("DKB")
    except DataClayException:
        DKB().make_persistent("DKB")

    start_time = time.time()
    # execute_trackers(["192.168.50.103", "192.168.50.103:5558", "192.168.50.103:5557"])
    # execute_trackers(["192.168.50.103", "192.168.50.103:5558"])
    execute_trackers(["192.168.50.103"]) 
    compss_barrier()

    print("ExecTime: " + str(time.time() - start_time))

    print("Exiting Application...")
    finish()


if __name__ == "__main__":
    main()

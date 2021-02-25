from datetime import datetime
from pycompss.api.parameter import *
from pycompss.api.task import task
from pycompss.api.api import compss_barrier, compss_wait_on
from pycompss.api.constraint import constraint
from socket import timeout
from utils import pixel2GPS
import deduplicator as dd
import paho.mqtt.client as mqtt
import track
# import threading

NUM_ITERS = 60
SNAP_PER_FEDERATION = 15
N = 5
CD_PROC = 0


# @constraint(AppSoftware="nvidia")
@task(returns=3, list_boxes=IN, trackers=IN, cur_index=IN, init_point=IN)
def execute_tracking(list_boxes, trackers, cur_index, init_point):
    return track.track2(list_boxes, trackers, cur_index, init_point)


"""
# @constraint(AppSoftware="xavier")
@task(returns=5,)
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
    box_coords = []
    for offset in range(1 + int_size + unsigned_long_size, len(message), 
                        double_size * 10 + int_size + 1 + float_size * 4):
        north, east, frame_number, obj_class = struct.unpack_from('ddIc', message[
                                                                        offset:offset + double_size * 2 + int_size + 1])
        x, y, w, h = struct.unpack_from('ffff', message[offset + double_size * 2 + int_size + 1:offset + double_size * 2
                                                                        + int_size + 1 + float_size * 4])
        # print((coord_north, coord_east, frame_number, ord(obj_class), x, y, w, h))
        boxes.append(track.obj_m(north, east, frame_number, ord(obj_class), int(w), int(h), int(x), int(y), 0.0))
        lat_ur, lon_ur, lat_lr, lon_lr, lat_ll, lon_ll, lat_ul, lon_ul = struct.unpack_from('dddddddd', message[
                                                                        offset + double_size * 2 + int_size + 1 +
                                                                        float_size * 4:])
        box_coords.append((lat_ur, lon_ur, lat_lr, lon_lr, lat_ll, lon_ll, lat_ul, lon_ul))
    return cam_id, timestamp, boxes, dummy, box_coords
"""


@constraint(AppSoftware="xavier")
@task(returns=6, )
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
    # pixels = []  # for logging purposes it is needed
    box_coords = []
    lat, lon = struct.unpack_from("dd", message[1 + int_size + unsigned_long_size:1 + int_size + unsigned_long_size
                                                                                  + double_size * 2])
    init_point = (lat, lon)
    for offset in range(1 + int_size + unsigned_long_size + double_size * 2, len(message),
                        double_size * 10 + int_size + 1 + float_size * 4):
        north, east, frame_number, obj_class = struct.unpack_from('ddIc', message[
                                                                        offset:offset + double_size * 2 + int_size + 1])
        x, y, w, h = struct.unpack_from('ffff', message[offset + double_size * 2 + int_size + 1:offset + double_size * 2
                                                                        + int_size + 1 + float_size * 4])
        boxes.append(track.obj_m(north, east, frame_number, ord(obj_class), int(w), int(h), int(x), int(y), 0.0))
        lat_ur, lon_ur, lat_lr, lon_lr, lat_ll, lon_ll, lat_ul, lon_ul = struct.unpack_from('dddddddd', message[
                                                                        offset + double_size * 2 + int_size + 1 +
                                                                        float_size * 4:])
        box_coords.append((lat_ur, lon_ur, lat_lr, lon_lr, lat_ll, lon_ll, lat_ul, lon_ul))
        # pixels.append((x, y))
    # return cam_id, timestamp, boxes, dummy # TODO: added x, y (pixels) as they are not in list_boxes anymore
    return cam_id, timestamp, boxes, dummy, box_coords, init_point


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
@task(trackers_list=COLLECTION_IN, cam_ids=COLLECTION_IN)  # , init_point=IN)
def deduplicate(trackers_list, cam_ids):  # init_point):
    return_message = dd.compute_deduplicator(trackers_list, cam_ids)
    # print(f"Returned {len(return_message)} objects (from the original "
    #      f"{' + '.join([str(len(t)) for t in trackers_list])} = {sum([len(t) for t in trackers_list])})")
    return return_message


def dump(id_cam, ts, trackers, iteration, list_boxes, info_for_deduplicator, box_coords):
    import pygeohash as pgh
    import os
    filename = "singlecamera.in"
    if not os.path.exists(filename):
        f = open(filename, "w+")
        f.close()
    with open(filename, "a+") as f:
        # for i, tracker in enumerate([t for t in trackers if t.traj[-1].frame == iteration]):
        idx = 0
        for i, tracker in enumerate(trackers):
            if tracker.id not in [t.id for t in trackers if t.traj[-1].frame == iteration]:
                continue
            lat = info_for_deduplicator[idx][0]  # round(info_for_deduplicator[i][0], 14)
            lon = info_for_deduplicator[idx][1]  # round(info_for_deduplicator[i][1], 14)
            geohash = pgh.encode(lat, lon, precision=7)
            cl = info_for_deduplicator[idx][2]
            speed = abs(tracker.ekf.xEst.vel)  # info_for_deduplicator[i][3]
            yaw = tracker.ekf.xEst.yaw  # info_for_deduplicator[i][4]
            pixel_x = info_for_deduplicator[idx][6]  # OR list_boxes[tracker.idx].x  # pixels[tracker.idx][0]
            pixel_y = info_for_deduplicator[idx][7]  # pixels[tracker.idx][1]
            f.write(
                # f"{id_cam} {iteration} {ts} {cl} {lat:.14f} {lon:.14f} {geohash} {speed} {yaw} {id_cam}_{tracker.id} \
                f"{id_cam} {iteration} {ts} {cl} {lat} {lon} {geohash} {speed} {yaw} {id_cam}_{tracker.id} {pixel_x} \
                {pixel_y} {list_boxes[tracker.idx].w} {list_boxes[tracker.idx].h} {boxCoords[tracker.idx][0]} \
                {boxCoords[tracker.idx][1]} {boxCoords[tracker.idx][2]} {boxCoords[tracker.idx][3]} \
                {boxCoords[tracker.idx][4]} {boxCoords[tracker.idx][5]} {boxCoords[tracker.idx][6]} \
                {boxCoords[tracker.idx][7]}\n")
            idx += 1


@constraint(AppSoftware="xavier")
@task(returns=1, trackers_list=COLLECTION_IN, count=IN, kb=IN)
def persist_info_accumulated(trackers_list, count, kb):
    from CityNS.classes import EventsSnapshot
    snapshot_alias = "events_" + str(count)
    snapshot = EventsSnapshot(snapshot_alias)
    kb.add_events_snapshot(snapshot) # persists snapshot
    for trackers in trackers_list:
        snapshot.add_events_from_trackers(trackers, kb) # create events inside dataclay
    return snapshot


@constraint(AppSoftware="xavier")
@task(returns=1, trackers=IN, count=IN, kb=IN)
def persist_info(trackers, count, kb):
    from CityNS.classes import EventsSnapshot
    snapshot_alias = "events_" + str(count)
    snapshot = EventsSnapshot(snapshot_alias)
    kb.add_events_snapshot(snapshot) # persists snapshot
    snapshot.add_events_from_trackers(trackers, kb)  # create events inside dataclay
    return snapshot


@task(snapshot=IN, dataclay_to_federate=IN)
def federate_info(snapshot, dataclay_to_federate):
    from CityNS.classes import FederationInfo
    snapshots = [snapshot]

    # create snapshots and events into FederateInfo structure
    federation_info = FederationInfo(snapshots)
    federation_info.make_persistent()

    # federate snapshots and events
    federation_info.federate(dataclay_to_federate)


@task(snapshots=COLLECTION_IN, dataclay_to_federate=IN)
def federate_info_accumulated(snapshots, dataclay_to_federate):
    from CityNS.classes import FederationInfo

    # create snapshots and events into FederateInfo structure
    federation_info = FederationInfo(snapshots)
    federation_info.make_persistent()

    # federate snapshots and events
    federation_info.federate(dataclay_to_federate)


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


# @task() # for my scheduler
@task(is_replicated=True)
def init_task():
    import uuid
    from CityNS.classes import DKB, Event, Object, EventsSnapshot, ListOfObjects, FederationInfo
    kb = DKB()
    kb.make_persistent("FAKE_" + str(uuid.uuid4()))
    kb.get_objects_from_dkb()
    snap = EventsSnapshot("FAKE_SNAP_" + str(uuid.uuid4()))
    snap.make_persistent("FAKE_SNAP_" + str(uuid.uuid4()))
    snap.get_objects_refs()
    event = Event(None, None, None, None, None, None, None)
    event.make_persistent("FAKE_EVENT_" + str(uuid.uuid4()))
    obj = Object("FAKE_OBJ_" + str(uuid.uuid4()), "FAKE", 0, 0, 0, 0)
    obj.make_persistent("FAKE_OBJ_" + str(uuid.uuid4()))
    obj.get_events_history(20)
    list_objs = ListOfObjects()
    list_objs.make_persistent("FAKE_LISTOBJ_" + str(uuid.uuid4()))
    list_objs.get_or_create("FAKE_LISTOBJ_" + str(uuid.uuid4()), "FAKE", 0, 0, 0, 0)
    federation_info = FederationInfo([snap])
    federation_info.make_persistent()
    federation_info.objects_per_snapshot # to load dataclay class and libraries


@constraint(AppSoftware="nvidia")
@task(returns=3, trackers_list=IN, tracker_indexes=IN, cur_index=IN)
def boxes_and_track(socket_ip, trackers_list, tracker_indexes, cur_index):
    _, _, list_boxes, _ = receive_boxes(socket_ip, 0)
    return execute_tracking(list_boxes, trackers_list, tracker_indexes, cur_index)


def execute_trackers(pipe_paths, kb):
    import uuid
    import time
    import sys
    import os
    from dataclay.api import get_dataclay_id

    trackers_list = [[]] * len(pipe_paths)
    cur_index = [0] * len(pipe_paths)
    info_for_deduplicator = [0] * len(pipe_paths)
    snapshots = list()  # accumulate snapshots
    cam_ids = [0] * len(pipe_paths)
    timestamps = [0] * len(pipe_paths)
    deduplicated_trackers_list = []  # TODO: accumulate trackers
    box_coords = [0] * len(pipe_paths)

    federation_ip, federation_port = "192.168.7.32", 11034  # TODO: change port accordingly
    # federation_ip, federation_port = "192.168.50.103", 21034 # TODO: change port accordingly
    dataclay_to_federate = get_dataclay_id(federation_ip, federation_port)

    i = 0
    reception_dummies = [0] * len(pipe_paths)
    start_time = time.time()
    while i < NUM_ITERS:
        for index, pipe_path in enumerate(pipe_paths):
            # cam_ids[index], timestamps[index], list_boxes, reception_dummies[index], pixels[index], sizes[index] = \
            cam_ids[index], timestamps[index], list_boxes, reception_dummies[index], box_coords[index], init_point = \
                receive_boxes(pipe_path, reception_dummies[index])
            trackers_list[index], cur_index[index], info_for_deduplicator[index] = execute_tracking(list_boxes,
                                                                                                    trackers_list[index],
                                                                                                    cur_index[index],
                                                                                                    init_point)
            # print(f"CAM ID: {cam_ids[index]}, timestamp: {timestamps[index]}, list_boxes: {[list_boxes]}")
            # dump(cam_ids[index], timestamps[index], trackers_list[index], i, list_boxes, \
            # info_for_deduplicator[index]), box_coords[index])  #, pixels[index])

        # trackers, tracker_indexes, cur_index = merge_tracker_state(trackers_list)
        deduplicated_trackers = deduplicate(info_for_deduplicator, cam_ids) # , init_point) # pass init point for gc
        # deduplicated_trackers_list.append(deduplicated_trackers) # TODO: accumulate trackers
        """
        for trackers in trackers_list:
            for idx, tracker in enumerate(trackers):
                if tracker.id not in [t.id for t in trackers_list[0] if t.traj[-1].frame == i]:
                    continue
                cl = info_for_deduplicator[0][idx][2]
                vel = info_for_deduplicator[0][idx][3]
                yaw = info_for_deduplicator[0][idx][4]
                lat = info_for_deduplicator[0][idx][0] # round(info_for_deduplicator[0][idx][0], 14)
                lon = info_for_deduplicator[0][idx][1] # round(info_for_deduplicator[0][idx][1], 14)
                track_id = info_for_deduplicator[0][idx][5]
                pixel_x = info_for_deduplicator[0][idx][6]
                pixel_y = info_for_deduplicator[0][idx][7]
                pixel_w = info_for_deduplicator[0][idx][8]
                pixel_h = info_for_deduplicator[0][idx][9]
                return_dedu.append((cam_ids[0], tracker.id, cl, vel, yaw, lat, lon, pixel_x, pixel_y, pixel_w, pixel_h))
                # return_dedu[idx] = tuple(j for i in return_dedu[idx] for j in (i if isinstance(i, tuple) else (i,)))
        deduplicated_trackers = (timestamps[0], return_dedu)
        """

        """# TODO: accumulate trackers
        if i != 0 and (i+1) % N == 0:
            snapshot = persist_info_accumulated(deduplicated_trackers_list, i, kb)
            deduplicated_trackers_list.clear() 
        """
        snapshot = persist_info(deduplicated_trackers, i, kb)
        """
        snapshots.append(snapshot)
        if i != 0 and (i+1) % SNAP_PER_FEDERATION == 0:
            federate_info_accumulated(snapshots, kb, dataclay_to_federate)
        """
        federate_info(snapshot, kb, dataclay_to_federate)
        i += 1
        if i != 0 and i % 10 == 0:
            compss_barrier()
            input_path = "/home/nvidia/CLASS/class-app/phemlight/in/"
            output_file = "results_" + str(uuid.uuid4()).split("-")[-1] + ".csv"
            analyze_pollution(input_path, output_file)

    compss_barrier()
    end_time = time.time()
    print("Exec Inner Time: " + str(end_time - start_time))
    print("Exec Inner Time per Iteration: " + str((end_time - start_time) / NUM_ITERS))


def on_message(client, userdata, message):
    import time
    global CD_PROC
    CD_PROC += 1
    received_time = time.time()
    msg = str(message.payload.decode('utf-8'))
    print(f"Received message = \"{msg}\" at time {received_time}")
    f = open("cd_log.txt", "a")
    f.write(msg)
    f.close()


def publish_mqtt(client):
    client.publish("test", "Start of the execution of the COMPSs workflow")


def register_mqtt():
    client = mqtt.Client()
    try:
        client.connect("192.168.7.42")  # MQTT server in Modena cloud
    except timeout as e:
        print(e)
        print("VPN Connection not active. Needed for MQTT.")
        exit()
    client.on_message=on_message
    client.subscribe("test")
    client.subscribe("tp-out")
    client.subscribe("cd-out")
    return client


def main():
    import sys
    import time
    from dataclay.api import init, finish
    from dataclay.exceptions.exceptions import DataClayException

    mqtt_wait = False
    if len(sys.argv) == 2:
        mqtt_wait = (sys.argv[1] != "False")

    init()
    from CityNS.classes import DKB, ListOfObjects

    # Register MQTT client to subscribe to MQTT server in 192.168.7.42
    if mqtt_wait:
        client = register_mqtt()
        client.loop_start()

    # initialize all computing units in all workers
    num_cus = 8
    for i in range(num_cus):
        init_task()
    compss_barrier()

    # Publish to the MQTT broker that the execution has started
    if mqtt_wait:
        publish_mqtt(client)

    try:
        kb = DKB.get_by_alias("DKB")
    except DataClayException:
        kb = DKB()
        list_objects = ListOfObjects()
        list_objects.make_persistent()
        kb.list_objects = list_objects
        kb.make_persistent("DKB")

    start_time = time.time()
    # execute_trackers(["192.168.50.103"], kb)
    execute_trackers([("/tmp/pipe_yolo2COMPSs", "/tmp/pipe_COMPSs2yolo")], kb)
    # pipe_paths = [("/tmp/pipe_yolo2COMPSs", "/tmp/pipe_COMPSs2yolo"), ("/tmp/pipe_write",  "/tmp/pipe_read")]
    # print("ExecTime: " + str(time.time() - start_time))
    # print("ExecTime per Iteration: " + str((time.time() - start_time) / NUM_ITERS))

    if mqtt_wait:
        while CD_PROC < NUM_ITERS:
            pass
    print("Exiting Application...")
    finish()


if __name__ == "__main__":
    main()

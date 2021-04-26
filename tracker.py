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
import socket
# import threading

NUM_ITERS = 400
SNAP_PER_FEDERATION = 15
N = 5
NUM_ITERS_FOR_CLEANING = 30
CD_PROC = 0
# mqtt_wait = True


# @constraint(AppSoftware="nvidia")
@task(returns=3, list_boxes=IN, trackers=IN, cur_index=IN, init_point=IN)
def execute_tracking(list_boxes, trackers, cur_index, init_point):
    return track.track2(list_boxes, trackers, cur_index, init_point)



# @constraint(AppSoftware="xavier")
@task(returns=6,)
def receive_boxes(socket_ip, dummy):
    import zmq
    import struct
    import time
    import traceback

    if ":" not in socket_ip:
        socket_ip += ":5559"
    
    message = b""
    cam_id = None   
    timestamp = None
    boxes = None    
    box_coords = None
    init_point = None
    no_read = True

    context = zmq.Context()
    sink = context.socket(zmq.REP)
    sink.connect(f"tcp://{socket_ip}")

    double_size = unsigned_long_size = 8
    int_size = float_size = 4
    boxes = []

    while no_read:
        try:
            no_read = False
            #time.sleep(0.05)
            message = sink.recv(zmq.NOBLOCK)
            sink.send_string("", zmq.NOBLOCK) 
            
            flag = len(message) > 0
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
        except zmq.ZMQError as e:
            no_read = True
            # traceback.print_exc()
            if e.errno == zmq.EAGAIN:
                pass
            else:
                traceback.print_exc()
    
    return cam_id, timestamp, boxes, dummy, box_coords, init_point


# @constraint(AppSoftware="xavier")
# @task(returns=5, )
# def receive_boxes(pipe_paths, dummy):
#     import struct

#     double_size = unsigned_long_size = 8
#     int_size = float_size = 4

#     # opening and closing pipes at each task otherwise read gets blocked until no writer
#     fifo_read = open(pipe_paths[0], 'rb')
#     fifo_write = open(pipe_paths[1], 'w')

#     boxes = []
#     message = fifo_read.read()
#     fifo_read.close()
#     fifo_write.write('0')
#     fifo_write.close()
#     flag = (message == 0)
#     # This flag serves to know if the video has ended
#     cam_id = struct.unpack_from("i", message[1:1 + int_size])[0]
#     timestamp = struct.unpack_from("Q", message[1 + int_size:1 + int_size + unsigned_long_size])[0]
#     # pixels = []  # for logging purposes it is needed
#     box_coords = []
#     for offset in range(1 + int_size + unsigned_long_size, len(message),
#                         double_size * 10 + int_size + 1 + float_size * 4):
#         north, east, frame_number, obj_class = struct.unpack_from('ddIc', message[
#                                                                         offset:offset + double_size * 2 + int_size + 1])
#         x, y, w, h = struct.unpack_from('ffff', message[offset + double_size * 2 + int_size + 1:offset + double_size * 2
#                                                                         + int_size + 1 + float_size * 4])
#         boxes.append(track.obj_m(north, east, frame_number, ord(obj_class), int(w), int(h), int(x), int(y), 0.0))
#         lat_ur, lon_ur, lat_lr, lon_lr, lat_ll, lon_ll, lat_ul, lon_ul = struct.unpack_from('dddddddd', message[
#                                                                         offset + double_size * 2 + int_size + 1 +
#                                                                         float_size * 4:])
#         box_coords.append((lat_ur, lon_ur, lat_lr, lon_lr, lat_ll, lon_ll, lat_ul, lon_ul))
#         # pixels.append((x, y))
#     # return cam_id, timestamp, boxes, dummy # TODO: added x, y (pixels) as they are not in list_boxes anymore
#     return cam_id, timestamp, boxes, dummy, box_coords


# @task(returns=3, )
# def merge_tracker_state(trackers_list, cur_index):
#     import itertools
#     trackers = []
#     prev_cur_index = cur_index
#     for tracker in itertools.chain.from_iterable(trackers_list):
#         if tracker.id >= prev_cur_index:
#             tracker.id = cur_index
#             cur_index += 1
#         trackers.append(tracker)
#     tracker_indexes = [True] * cur_index + [False] * (32767 - cur_index)
#     return trackers, tracker_indexes, cur_index


# @constraint(AppSoftware="nvidia")
@task(trackers_list=COLLECTION_IN, cam_ids=COLLECTION_IN)
# def deduplicate(trackers_list, cam_ids, timestamps): # TODO: waiting for UNIMORE
def deduplicate(trackers_list, cam_ids):
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
        for i, tracker in enumerate(trackers):
            if tracker.id not in [t.id for t in trackers if t.traj[-1].frame == iteration]:
                continue
            lat = info_for_deduplicator[i][0]  # round(info_for_deduplicator[i][0], 14)
            lon = info_for_deduplicator[i][1]  # round(info_for_deduplicator[i][1], 14)
            geohash = pgh.encode(lat, lon, precision=7)
            cl = info_for_deduplicator[i][2]
            speed = abs(tracker.ekf.xEst.vel)  # info_for_deduplicator[i][3]
            yaw = tracker.ekf.xEst.yaw  # info_for_deduplicator[i][4]
            pixel_x = info_for_deduplicator[i][6]  # OR list_boxes[tracker.idx].x  # pixels[tracker.idx][0]
            pixel_y = info_for_deduplicator[i][7]  # pixels[tracker.idx][1]
            f.write(
                # f"{id_cam} {iteration} {ts} {cl} {lat:.14f} {lon:.14f} {geohash} {speed} {yaw} {id_cam}_{tracker.id} \
                f"{id_cam} {iteration} {ts} {cl} {lat} {lon} {geohash} {speed} {yaw} {id_cam}_{tracker.id} {pixel_x} \
                {pixel_y} {list_boxes[tracker.idx].w} {list_boxes[tracker.idx].h} {box_coords[tracker.idx][0]} \
                {box_coords[tracker.idx][1]} {box_coords[tracker.idx][2]} {box_coords[tracker.idx][3]} \
                {box_coords[tracker.idx][4]} {box_coords[tracker.idx][5]} {box_coords[tracker.idx][6]} \
                {box_coords[tracker.idx][7]}\n")


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
    snapshot.make_persistent()
    snapshot.add_events_from_trackers(trackers, kb)  # create events inside dataclay
    return snapshot


@constraint(AppSoftware="xavier")
@task(snapshot=IN, backend_to_federate=IN)
def federate_info(snapshot, backend_to_federate):
    snapshot.federate_to_backend(backend_to_federate)


@task(snapshots=COLLECTION_IN, backend_to_federate=IN)
def federate_info_accumulated(snapshots, backend_to_federate):
    for snapshot in snapshots:
        snapshot.federate_to_backend(backend_to_federate)


@task(kb=IN, foo=INOUT)
def remove_objects_from_dataclay(kb, foo):
    # from dataclay.api import get_num_objects
    kb.remove_old_snapshots_and_objects(int(datetime.now().timestamp() * 1000), True)
    # print(f"Current number of objects in dataclay: {get_num_objects()}")
    return foo


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
    from CityNS.classes import DKB, Event, Object, EventsSnapshot
    kb = DKB()
    kb.make_persistent("FAKE_" + str(uuid.uuid4()))
    # kb.get_objects_from_dkb()
    snap = EventsSnapshot("FAKE_SNAP_" + str(uuid.uuid4()))
    snap.make_persistent("FAKE_SNAP_" + str(uuid.uuid4()))
    snap.snap_alias
    event = Event(None, None, None, None, None, None, None)
    event.make_persistent("FAKE_EVENT_" + str(uuid.uuid4()))
    obj = Object("FAKE_OBJ_" + str(uuid.uuid4()), "FAKE", 0, 0, 0, 0)
    obj.make_persistent("FAKE_OBJ_" + str(uuid.uuid4()))
    obj.get_events_history(20)


@constraint(AppSoftware="nvidia")
@task(returns=3, trackers_list=IN, tracker_indexes=IN, cur_index=IN)
def boxes_and_track(socket_ip, trackers_list, tracker_indexes, cur_index):
    _, _, list_boxes, _ = receive_boxes(socket_ip, 0)
    return execute_tracking(list_boxes, trackers_list, tracker_indexes, cur_index)


def execute_trackers(socket_ips, kb):
    import uuid
    import time
    import sys
    import os
    from dataclay.api import register_dataclay, get_external_backend_id_by_name

    trackers_list = [[]] * len(socket_ips)
    cur_index = [0] * len(socket_ips)
    info_for_deduplicator = [0] * len(socket_ips)
    snapshots = list()  # accumulate snapshots
    cam_ids = [0] * len(socket_ips)
    timestamps = [0] * len(socket_ips)
    deduplicated_trackers_list = []  # TODO: accumulate trackers
    box_coords = [0] * len(socket_ips)

    federation_ip, federation_port = "192.168.7.32", 11034  # TODO: change port accordingly
    # federation_ip, federation_port = "192.168.50.103", 21034 # TODO: change port accordingly
    dataclay_to_federate = register_dataclay(federation_ip, federation_port)
    external_backend_id = get_external_backend_id_by_name("DS1", dataclay_to_federate)

    i = 0
    reception_dummies = [0] * len(socket_ips)
    start_time = time.time()
    foo = None
    while i < NUM_ITERS:
        for index, socket_ip in enumerate(socket_ips):
            # cam_ids[index], timestamps[index], list_boxes, reception_dummies[index], pixels[index], sizes[index] = \
            cam_ids[index], timestamps[index], list_boxes, reception_dummies[index], box_coords[index], init_point = \
                receive_boxes(socket_ip, reception_dummies[index])
            trackers_list[index], cur_index[index], info_for_deduplicator[index] = execute_tracking(list_boxes,
                                                                                                    trackers_list[index],
                                                                                                    cur_index[index],
                                                                                                    init_point)
            # print(f"CAM ID: {cam_ids[index]}, timestamp: {timestamps[index]}, list_boxes: {[list_boxes]}")
            # dump(cam_ids[index], timestamps[index], trackers_list[index], i, list_boxes, \
            # info_for_deduplicator[index]), box_coords[index])  #, pixels[index])

        # trackers, tracker_indexes, cur_index = merge_tracker_state(trackers_list)
        deduplicated_trackers = deduplicate(info_for_deduplicator, cam_ids) # , cam_ids, timestamps) # pass cam_ids and timestamp
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
            federate_info_accumulated(snapshots, dataclay_to_federate)
        """
        federate_info(snapshot, external_backend_id)
        i += 1

        if i != 0 and i % NUM_ITERS_FOR_CLEANING == 0:
            # delete objects based on timestamps
            # kb.remove_old_snapshots_and_objects(int(datetime.now().timestamp() * 1000), True)
            foo = remove_objects_from_dataclay(kb, foo)
            # compss_barrier()
            # input_path = "/home/nvidia/CLASS/class-app/phemlight/in/"
            # output_file = "results_" + str(uuid.uuid4()).split("-")[-1] + ".csv"
            # analyze_pollution(input_path, output_file)

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


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    import sys
    import time
    from dataclay.api import init, finish
    from dataclay.exceptions.exceptions import DataClayException
    import argparse

    # if len(sys.argv) != 2:
    #     print("Incorrect number of params: python3 tracker.py ${TKDNN_IP} ${MQTT_ACTIVE} (optional)")
    # tkdnn_ip = sys.argv[1]
    # if len(sys.argv) == 3:
    #     mqtt_wait = (sys.argv[2] != "False")

    # Parse arguments to accept variable number of "IPs:Ports"
    parser = argparse.ArgumentParser()
    parser.add_argument("tkdnn_ips", nargs='+')
    parser.add_argument("mqtt_wait", nargs='?', const=True, type=str2bool, default=False)  # True as default
    args = parser.parse_args()
    
    init()
    from CityNS.classes import DKB

    # Register MQTT client to subscribe to MQTT server in 192.168.7.42
    if args.mqtt_wait:
        client = register_mqtt()
        client.loop_start()

    # initialize all computing units in all workers
    num_cus = 8
    for i in range(num_cus):
        init_task()
    compss_barrier()

    # Publish to the MQTT broker that the execution has started
    if args.mqtt_wait:
        publish_mqtt(client)

    try:
        kb = DKB.get_by_alias("DKB")
    except DataClayException:
        kb = DKB()
        kb.make_persistent("DKB")
    # kb = None
    start_time = time.time()
    execute_trackers(args.tkdnn_ips, kb)

    """
    if args.mqtt_wait:
        while CD_PROC < NUM_ITERS:
            pass
    """

    print("Exiting Application...")
    finish()


if __name__ == "__main__":
    main()

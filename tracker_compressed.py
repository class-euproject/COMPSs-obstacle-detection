import datetime
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

NUM_ITERS = 800
NUM_ITERS_POLLUTION = 25
SNAP_PER_FEDERATION = 15
N = 5
NUM_ITERS_FOR_CLEANING = 400
CD_PROC = 0

pollution_file_name = "pollution.csv"

# @constraint(AppSoftware="nvidia")
@task(returns=3, list_boxes=IN, trackers=IN, cur_index=IN, init_point=IN)
def execute_tracking(list_boxes, trackers, cur_index, init_point):
    return track.track2(list_boxes, trackers, cur_index, init_point)


# @constraint(AppSoftware="xavier")
@task(returns=7,)
def receive_boxes(socket_ip, dummy):
    import struct
    import time
    import traceback

    socket_port = 5559
    if ":" in socket_ip:
        socket_ip, socket_port = socket_ip.split(":")
        socket_port = int(socket_port)
    
    message = b""
    cam_id = None   
    timestamp = None
    boxes = None    
    box_coords = None
    init_point = None
    no_read = True
    frame_number = -1

    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverSocket.sendto(b"A", (socket_ip, socket_port))

    double_size = unsigned_long_size = 8
    int_size = float_size = 4
    boxes = []

    while no_read:
        try:
            no_read = False
            message, address = serverSocket.recvfrom(16000)

            flag = len(message) > 0
            # This flag serves to know if the video has ended
            cam_id = struct.unpack_from("i", message[1:1 + int_size])[0]
            timestamp = struct.unpack_from("Q", message[1 + int_size:1 + int_size + unsigned_long_size])[0]
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
        except socket.error:
            no_read = True
            traceback.print_exc()

    return cam_id, timestamp, boxes, dummy, box_coords, init_point, frame_number


@constraint(AppSoftware="xavier")
@task(trackers_list=COLLECTION_IN, cam_ids=COLLECTION_IN, foo_dedu=INOUT, frames=COLLECTION_IN)
def deduplicate(trackers_list, cam_ids, foo_dedu, frames):
    return_message = dd.compute_deduplicator(trackers_list, cam_ids, frames)
    return return_message, foo_dedu


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
    snapshot = EventsSnapshot(snapshot_alias, print_federated_events=True, trigger_modena_tp=False)
    snapshot.make_persistent()
    for trackers in trackers_list:
        snapshot.add_events_from_trackers(trackers, kb)  # create events inside dataclay
    return snapshot


@constraint(AppSoftware="xavier")
@task(returns=1, trackers=IN, count=IN, kb=IN, with_pollution=IN)
def persist_info(trackers, count, kb, with_pollution):
    classes = ["person", "car", "truck", "bus", "motor", "bike", "rider", "traffic light", "traffic sign", "train",
               "class11", "class12", "class13", "class14", "class15", "class16", "class17", "class18","class19",
               "class20", "class21", "class22", "class23", "class24", "class25", "class26", "class27", "class28",
               "class29", "class30", "class31"]
    from CityNS.classes import EventsSnapshot
    snapshot_alias = "events_" + str(count)
    snapshot = EventsSnapshot(snapshot_alias, print_federated_events=True, trigger_modena_tp=False)
    snapshot.make_persistent()
    snapshot.add_events_from_trackers(trackers, kb)  # create events inside dataclay
    if with_pollution:
        if not os.path.exists(pollution_file_name):
            with open(pollution_file_name, "w") as f:
                f.write("VehID, LinkID, Time, Vehicle_type, Av_link_speed\n")
        for index, ev in enumerate(trackers[1]):
            if classes[ev[2]] in ["car", "bus"]:
                obj_type = (classes[ev[2]]).title()
            elif classes[ev[2]] in ["class20", "class30", "class31"]:
                obj_type = "car"
            elif classes[ev[2]] == "truck":
                obj_type = "HDV"
            else:
                continue
            with open(pollution_file_name, "a") as f:
                f.write(f"{ev[0]}_{ev[1]}, {ev[0]}, {trackers[0]}, {obj_type}, {ev[3]}\n")
    return snapshot


@constraint(AppSoftware="xavier")
@task(kb=IN, snapshot=IN, backend_to_federate=IN)
def federate_info(kb, snapshot, backend_to_federate):
    kb.federate_compressed_snapshot(snapshot, backend_to_federate)


@task(kb=IN, snapshots=COLLECTION_IN, backend_to_federate=IN)
def federate_info_accumulated(kb, snapshots, backend_to_federate):
    for snapshot in snapshots:
        kb.federate_compressed_snapshot(snapshot, backend_to_federate)


@task(kb=IN, foo=INOUT)
def remove_objects_from_dataclay(kb, foo):
    # Remove snapshots and objects older than 10 minutes
    kb.remove_old_snapshots_and_objects(
        int((datetime.datetime.now() - datetime.timedelta(minutes=10)).timestamp() * 1000), True)
    return foo


# @constraint(AppSoftware="phemlight") # TODO: to be executed in Cloud. Remove it otherwise
@constraint(AppSoftware="xavier")
@task(foo=INOUT)
def analyze_pollution(foo):
    a = 'oquesea'
    b = pollution_file_name
    c = 'stream.csv'
    os.system(f"scp {pollution_file_name} esabate@192.168.7.32:/m/home/esabate/pollutionMap/phemlight-r/in/ && rm {pollution_file_name}")
    os.system(f"ssh esabate@192.168.7.32 nohup bash dockerScripts/phemlightCommand.sh  {a} {b} {c} &>/dev/null & ")
    return foo


# @task() # for my scheduler
@task(is_replicated=True)
def init_task():
    import uuid
    from CityNS.classes import DKB, Event, Object, EventsSnapshot, FederationCompressedInfo
    kb = DKB()
    kb.make_persistent("FAKE_" + str(uuid.uuid4()))
    # kb.get_objects_from_dkb()
    snap = EventsSnapshot("FAKE_SNAP_" + str(uuid.uuid4()))
    snap.make_persistent("FAKE_SNAP_" + str(uuid.uuid4()))
    snap.snap_alias
    event = Event()
    event.make_persistent("FAKE_EVENT_" + str(uuid.uuid4()))
    obj = Object("FAKE_OBJ_" + str(uuid.uuid4()), "FAKE")
    obj.make_persistent("FAKE_OBJ_" + str(uuid.uuid4()))
    obj.get_events_history(20)
    compressed = FederationCompressedInfo(snap, kb)
    compressed.snap_alias


@constraint(AppSoftware="nvidia")
@task(returns=3, trackers_list=IN, tracker_indexes=IN, cur_index=IN)
def boxes_and_track(socket_ip, trackers_list, tracker_indexes, cur_index):
    _, _, list_boxes, _ = receive_boxes(socket_ip, 0)
    return execute_tracking(list_boxes, trackers_list, tracker_indexes, cur_index)


def execute_trackers(socket_ips, kb, with_pollution):
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
    frames = [0] * len(socket_ips)

    federation_ip, federation_port = "192.168.7.32", 11034  # TODO: change port accordingly
    dataclay_to_federate = register_dataclay(federation_ip, federation_port)
    external_backend_id = get_external_backend_id_by_name("DS1", dataclay_to_federate)

    i = 0
    reception_dummies = [0] * len(socket_ips)
    start_time = time.time()
    foo_dedu = foo = None
    while i < NUM_ITERS:
        for index, socket_ip in enumerate(socket_ips):
            cam_ids[index], timestamps[index], list_boxes, reception_dummies[index], box_coords[index], init_point, frames[index] = \
                receive_boxes(socket_ip, reception_dummies[index])
            trackers_list[index], cur_index[index], info_for_deduplicator[index] = execute_tracking(list_boxes,
                                                                                                    trackers_list[index],
                                                                                                    cur_index[index],
                                                                                                    init_point)

        deduplicated_trackers, foo_dedu = deduplicate(info_for_deduplicator, cam_ids, foo_dedu, frames)  # or frames appended inside info_for_dedu in tracking

        """# TODO: accumulate trackers
        if i != 0 and (i+1) % N == 0:
            snapshot = persist_info_accumulated(deduplicated_trackers_list, i, kb)
            deduplicated_trackers_list.clear() 
        """
        # frames[0] not correct when more than one camera. It should be passed in info_for_deduplicator and returned in deduplicated_trackers
        snapshot = persist_info(deduplicated_trackers, kb, with_pollution)
        """
        snapshots.append(snapshot)
        if i != 0 and (i+1) % SNAP_PER_FEDERATION == 0:
            federate_info_accumulated(snapshots, dataclay_to_federate)
        """
        federate_info(kb, snapshot, external_backend_id)
        i += 1
        if i != 0 and i % NUM_ITERS_POLLUTION == 0 and with_pollution:
            print("Executing pollution")
            foo = analyze_pollution(foo)
        if i % NUM_ITERS_FOR_CLEANING == 0:
            # delete objects based on timestamps
            foo = remove_objects_from_dataclay(kb, foo)
        """
        if i % NUM_ITERS == 0:
            # compss_barrier()
            # analyze_pollution(foo)
        """

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
    import argparse	
    import sys
    import time
    import zmq
    from dataclay.api import init, finish
    from dataclay.exceptions.exceptions import DataClayException
		
    # Parse arguments to accept variable number of "IPs:Ports"	
    parser = argparse.ArgumentParser()
    parser.add_argument("tkdnn_ips", nargs='+')
    parser.add_argument("--mqtt_wait", nargs='?', const=True, type=str2bool, default=False)  # True as default
    parser.add_argument("--with_pollution", nargs='?', const=True, type=str2bool, default=False)  # True as default
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
    print(f"Init task completed {datetime.now()}")
    input("Press enter to continue...")

    # Publish to the MQTT broker that the execution has started
    if args.mqtt_wait:
        publish_mqtt(client)

    # Dataclay KB generation
    try:
        kb = DKB.get_by_alias("DKB")
    except DataClayException:
        kb = DKB()
        list_objects = ListOfObjects()
        list_objects.make_persistent()
        kb.list_objects = list_objects
        kb.make_persistent("DKB")

    ### ACK TO START WORKFLOW AT tkDNN ###
    for socket_ip in args.tkdnn_ips:
        if ":" not in socket_ip:
            socket_ip += ":5559"
        context = zmq.Context()
        sink = context.socket(zmq.REQ)
        sink.connect(f"tcp://{socket_ip}")
        sink.send_string("")
        sink.close()
        context.term()

    execute_trackers(args.tkdnn_ips, kb, args.with_pollution)

    if args.mqtt_wait:
        while CD_PROC < NUM_ITERS:
            pass

    print("Exiting Application...")
    finish()


if __name__ == "__main__":
    main()


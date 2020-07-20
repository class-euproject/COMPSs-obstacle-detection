from pycompss.api.parameter import *
from pycompss.api.task import task
from pycompss.api.api import compss_barrier, compss_wait_on
from collections import deque
from typing import Tuple, List
import track
from model import EventsSnapshot, Vehicle, VehicleType, Event, PedestrianFlow, Position
from datetime import datetime

QUAD_REG_LEN = 20
QUAD_REG_OFFSET = 5


class Boolean:
    def __init__(self, __val: bool):
        self.__value = __val

    def __eq__(self, other):
        if isinstance(other, Boolean):
            return self.__value == other.__value
        elif isinstance(other, bool):
            return self.__value == other
        else:
            return False

    def __str__(self):
        return str(self.__value)

    def set(self, __val):
        self.__value = __val


def traj_pred(dqx: deque, dqy: deque, dqt: deque) -> Tuple[List[float], List[float], List[float]]:
    import math
    vct_t = list(dqt)
    transformed_timestamps = []
    initial_timestamp = math.floor(vct_t[0] / 1000)
    for actual_timestamp in vct_t:
        transformed_timestamps.append(actual_timestamp / 1000 - initial_timestamp)
    vct_x = transformed_timestamps

    vct_y = list(dqx)
    vct_xp = list()
    ft = list()
    last_t = vct_x[-1]
    for i in range(1, QUAD_REG_OFFSET + 1):
        vct_xp.append(last_t + i)
        ft.append(vct_t[-1] + i * 1000)

    fx = quad_reg(vct_x, vct_y, vct_xp)

    vct_x = list(dqx)
    vct_y = list(dqy)

    if min(vct_x) <= fx[-1] <= max(vct_x):
        fy = circle_fit(vct_x, vct_y, fx)
    else:
        fy = quad_reg(vct_x, vct_y, fx)

    return fx, fy, ft


def circle_fit(vct_x: List, vct_y: List, xin: List) -> List[float]:
    import numpy as np
    import math

    x_bar = sum(vct_x) / len(vct_x)
    y_bar = sum(vct_y) / len(vct_y)

    u = [vct_x[i] - x_bar for i in range(len(vct_x))]
    v = [vct_y[i] - y_bar for i in range(len(vct_y))]

    # if all values are equal, return the same x,y coordinates
    if (len(np.unique(u)) == 1) or (len(np.unique(v)) == 1):
        fy = list()
        for x in xin:
            fy.append(vct_x[-1])
        return fy

    S_uu = S_vv = S_uuu = S_vvv = S_uv = S_uvv = S_vuu = 0.0

    for i in range(len(vct_x)):
        S_uu += u[i] * u[i]
        S_vv += v[i] * v[i]
        S_uuu += u[i] * u[i] * u[i]
        S_vvv += v[i] * v[i] * v[i]
        S_uv += u[i] * v[i]
        S_uvv += u[i] * v[i] * v[i]
        S_vuu += v[i] * u[i] * u[i]

    v_c = (S_uv * (S_uuu + S_uvv) - S_uu * (S_vvv + S_vuu)) / (2 * (S_uv * S_uv - S_uu * S_vv))
    u_c = (0.5 * (S_uuu + S_uvv) - v_c * S_uv) / S_uu
    x_c = u_c + x_bar
    y_c = v_c + y_bar

    a = u_c * u_c + v_c * v_c + (S_uu + S_vv) / len(vct_x)
    R = math.sqrt(a)

    # loop to predict multiple values
    fy = list()
    for x in xin:
        b = -2 * y_c
        xdiff = x - x_c
        c = y_c * y_c - R * R + xdiff * xdiff
        sr = b * b - 4 * c

        yout1 = yout2 = 0.0

        if sr < 0:
            fy.append(quad_reg(vct_x, vct_y, [x])[-1])  # get y from x
        else:
            yout1 = (-b + math.sqrt(b * b - 4 * c)) / 2
            yout2 = (-b - math.sqrt(b * b - 4 * c)) / 2

        if min(vct_y) <= yout1 <= max(vct_y):
            fy.append(yout1)
        else:
            fy.append(yout2)

    return fy


def quad_reg(vx: List, vy: List, z: List) -> List[float]:
    import numpy as np
    sum_xi2_by_yi = sum_xi_by_yi = sum_yi = sum_xi = sum_xi2 = sum_xi3 = sum_xi4 = 0.0

    for i in range(len(vx)):
        sum_xi += vx[i]
        sum_xi2 += vx[i] ** 2
        sum_xi3 += vx[i] ** 3
        sum_xi4 += vx[i] ** 4
        sum_yi += vy[i]
        sum_xi_by_yi += vx[i] * vy[i]
        sum_xi2_by_yi += vx[i] ** 2 * vy[i]

    A = np.array([[sum_xi4, sum_xi3, sum_xi2], [sum_xi3, sum_xi2, sum_xi], [sum_xi2, sum_xi, len(vx)]])
    b = np.array([sum_xi2_by_yi, sum_xi_by_yi, sum_yi])
    try:
        x_prime = np.linalg.solve(A, b)
    except np.linalg.linalg.LinAlgError:
        x_prime = np.linalg.lstsq(A, b)[0]  # for singular matrix exceptions

    a, b, c = x_prime[:3]

    # loop to predict multiple values
    fy = list()
    for x in z:
        fx = x
        fy.append(a * fx ** 2 + b * fx + c)

    return fy


@task(events=IN, returns=List[Tuple[float, float, float]])
def trajectory_pred_func(events: List[Event]) -> List[Tuple[float, float, float]]:
    dqx = deque()
    dqy = deque()
    dqt = deque()
    # print(f"events num: {len(events)}")
    # if len(events) <= QUAD_REG_LEN:
    #    return [(-1, -1, -1)]
    try:
        for event in events[-QUAD_REG_LEN:]:
            dqx.append(float(event.pos.lon))
            dqy.append(float(event.pos.lat))
            dqt.append(int(event.dt.timestamp()))
        # if len(dqx) > QUAD_REG_LEN:
        fx, fy, t = traj_pred(dqx, dqy, dqt)
        return [(fx[i], fy[i], t[i]) for i in range(len(fx))]
        # else:
        #     return [(-1, -1, -1)]
    except Exception as e:
        print(e)
        raise e
        # return -1, -1


# @constraint(AppSoftware="yolo")
@task(returns=list, listBoxes=IN, dt=IN, initial_age=IN, age_threshold=IN, trackers=IN)
def execute_tracking(listBoxes, dt, n_states, initial_age, age_threshold, trackers, fn):
    newBoxes = [t for t in listBoxes if fn(t)]
    return track.Track2(newBoxes, dt, n_states, initial_age, age_threshold, trackers)


# @constraint(AppSoftware="yolo")
@task(returns=Tuple[bool, list],)
def receive_boxes():
    import zmq
    import struct

    context = zmq.Context()
    sink = context.socket(zmq.REQ)
    sink.connect("tcp://127.0.0.1:5559")  # tcp://172.0.0.1 for containerized executions

    sink.send_string("")

    double_size = 8
    int_size = float_size = 4

    boxes = []
    print("Awaiting message...")
    message = sink.recv()
    flag = len(message) > 0
    print(f"Flag is {flag}")

    if flag:
        for offset in range(1, len(message), double_size * 2 + int_size + 1 + float_size * 4):
            coord_north, coord_east, frame_number, obj_class = struct.unpack_from('ddIc', message[
                                                                                          offset:offset + double_size * 2 + int_size + 1])
            x, y, w, h = struct.unpack_from('ffff', message[offset + double_size * 2 + int_size + 1:])
            # print((coord_north, coord_east, frame_number, ord(obj_class), x, y, w, h))
            boxes.append(track.Data(coord_north, coord_east, frame_number, ord(obj_class), x, y, w, h))
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


@task(returns=list, tracker=IN)
def populate_snapshot(tracker):
    events = []
    for ev in tracker:
        vel_pred = ev.pred_list_[-1].vel_ if len(ev.pred_list_) > 0 else -1
        lat = ev.traj_[-1].x_
        lon = ev.traj_[-1].y_
        if int(ev.class_) == 1:
            obj = Vehicle(VehicleType.Bicycle, float(vel_pred))
        elif int(ev.class_) == 5:
            obj = Vehicle(VehicleType.Bus, float(vel_pred))
        elif int(ev.class_) == 6:
            obj = Vehicle(VehicleType.Car, float(vel_pred))
        elif int(ev.class_) == 13:
            obj = Vehicle(VehicleType.Motorcycle, float(vel_pred))
        elif int(ev.class_) == 14:
            obj = PedestrianFlow(1)
        else:
            continue
        event = Event(obj, Position(float(lon), float(lat)), datetime.now(), ev.id_, ev.class_)
        events.append(event)
    return events


def execute_trackers():
    trackers1 = []
    trackers2 = []
    trackers3 = []

    initial_age = -5
    age_threshold = -8
    n_states = 5
    dt = 0.03

    video_resolution = (1920, 1080)  # TODO: RESOLUTION SHOULD NOT BE HARDCODED!
    reference_x, reference_y = [r // 2 for r in video_resolution]

    i = 0
    ret = True
    while ret:
        ret, list_boxes = compss_wait_on(receive_boxes())

        print(f"[{i}] Ret is {ret}")

        if ret:
            trackers1 = execute_tracking(list_boxes, dt, n_states, initial_age, age_threshold, trackers1,
                                         lambda t: t.x_pixel_ + t.w_ < reference_x and t.y_pixel_ + t.h_ < reference_y)
            trackers2 = execute_tracking(list_boxes, dt, n_states, initial_age, age_threshold, trackers2,
                                         lambda t: t.x_pixel_ + t.w_ >= reference_x and t.y_pixel_ + t.h_ < reference_y)
            trackers3 = execute_tracking(list_boxes, dt, n_states, initial_age, age_threshold, trackers3,
                                         lambda t: t.y_pixel_ + t.h_ >= reference_y)

            snapshot = EventsSnapshot()
            for tracker in [trackers1, trackers2, trackers3]:
                snapshot.add_events(compss_wait_on(populate_snapshot(tracker)))
            if i % 5 == 0 and i != 0:
                compss_barrier()

            i += 1
            compss_barrier()
            ids = compss_wait_on(snapshot.get_ids())
            for obj_id in ids:
                prediction = compss_wait_on(trajectory_pred_func(snapshot.get_by_id(obj_id).events))
                snapshot.add_prediction(obj_id, prediction)
                # print(f"Object Id={obj_id} Prediction={prediction}")


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

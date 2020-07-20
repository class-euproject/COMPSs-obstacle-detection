from typing import List, Dict, Union, KeysView, Tuple
from datetime import datetime
from enum import Enum


class Position:

    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.lon == other.lon and self.lat == other.lat

    def __hash__(self):
        return hash(self.lon) ^ hash(self.lat)

    def __str__(self):
        return 'Longitude: ' + str(self.lon) + ', Latitude: ' + str(self.lat)


VehicleType = Enum("VehicleType", "Bicycle Bus Car Motorcycle")


class Vehicle:

    def __init__(self, obj_type: VehicleType, speed: float, yaw: int = -1):
        self.type = obj_type
        self.speed = speed
        self.yaw = yaw

    def __str__(self):
        return 'Predicted speed: ' + str(self.speed) + ', Predicted Yaw: ' + str(self.yaw)


class PedestrianFlow:

    def __init__(self, value: int):
        self.value = value


class Event:

    def __init__(self, event_type: Union[Vehicle, PedestrianFlow], pos: Position, dtime: datetime, id_event: int, id_class: int):
        self.object = event_type
        self.pos = pos
        self.dt = dtime
        self.id_event = id_event
        self.id_class = id_class
        # print('Init Event ' + str(dtime))

    def __str__(self):
        return 'Event Type: ' + self.object.__class__.__name__ + ', Position: [' + str(
            self.pos) + '], Datetime: ' + str(self.dt) + ', idEvent: ' + str(self.id_event) + ', idClass: ' + str(
            self.id_class) + ' ' + str(self.object) + '\n'


class ObjMove:

    def __init__(self, size: int = 21):
        self.events: List[Event] = list()
        self.size: int = size

    def add_event(self, new_event: Event) -> None:
        self.events.append(new_event)
        if len(self.events) > self.size:
            self.events[:1] = []


class EventsSnapshot:

    def __init__(self, new_events: List[Event] = None):
        self.eventsById: Dict[int, ObjMove] = dict()
        self.tpById: Dict[int, List[Tuple[float, float, float]]] = dict()
        if new_events is not None:
            for e in new_events:
                ievents = self.eventsById.get(e.id_event)
                if ievents is None:
                    ievents = ObjMove()
                    self.eventsById[e.id_event] = ievents
                ievents.add_event(e)

    def get_by_id(self, oid: int) -> ObjMove:
        return self.eventsById[oid]

    def get_ids(self) -> KeysView:
        return self.eventsById.keys()

    def add_event(self, new_event: Event) -> None:
        ievents = self.eventsById.get(new_event.id_event)
        if ievents is None:
            ievents = ObjMove()
            self.eventsById[new_event.id_event] = ievents
        ievents.add_event(new_event)

    def add_prediction(self, oid: int, pred: List[Tuple[float, float, float]]) -> None:
        self.tpById[oid] = pred

    def get_predictions(self) -> Dict[int, List[Tuple[float, float, float]]]:
        return self.tpById

    def add_events(self, new_events: List[Event]) -> None:
        for e in new_events:
            ievents = self.eventsById.get(e.id_event)
            if ievents is None:
                ievents = ObjMove()
                self.eventsById[e.id_event] = ievents
            ievents.add_event(e)

    def delete(self):
        self.eventsById.clear()

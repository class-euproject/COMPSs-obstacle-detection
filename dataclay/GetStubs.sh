#!/bin/bash

docker run --rm -v $PWD/cfgfiles:/home/dataclayusr/dataclay/cfgfiles \
	   -v /compss/stubs:/stubs --network host \
	   bscdataclay/client:alpine GetStubs CityUser p4ssw0rd CityNS /stubs


rm /compss/dataclay.jar
ID=$(docker create bscdataclay/logicmodule:alpine)
docker cp $ID:/home/dataclayusr/dataclay/dataclay.jar /compss
docker rm $ID

#!/bin/bash

echo "Killing and removing dockers"
docker-compose kill && docker-compose down -v 

echo "Launching dockers"
docker-compose up -d

# check if dockers already up and registered
res=$(docker-compose logs 2>&1 | grep " ===> The ContractID for the registered classes is:")
while [ -z "$res" ];
do
        echo "Dataclay initializer not ready yet. Waiting for it to finish..."
        sleep 3
        res=$(docker-compose logs 2>&1 | grep " ===> The ContractID for the registered classes is:")
done
echo "Dataclay initializer registered model, stubs can be retrieved."

# Retrieving stubs
./GetStubs.sh

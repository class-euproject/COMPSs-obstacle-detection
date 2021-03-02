#!/bin/bash
# set -x
# set -e

CONTRACT_ID_FILE=${DC_SHARED_VOLUME}/${NAMESPACE}_contractid

########################### create cfgfiles ###########################

printf "HOST=${LOGICMODULE_HOST}\nTCPPORT=${LOGICMODULE_PORT_TCP}" > ${DATACLAYCLIENTCONFIG}
echo "Account=${USER}
Password=${PASS}
DataSets=${DATASET}
DataSetForStore=${DATASET}
StubsClasspath=${STUBSPATH}" > ${DATACLAYSESSIONCONFIG}

######################################################

# Wait for contract id in shared volume
while [ ! -f ${CONTRACT_ID_FILE} ]; do echo "Waiting for contract ID at ${CONTRACT_ID_FILE}..."; sleep 5; done

export CONTRACT_ID=`cat ${CONTRACT_ID_FILE}`

mkdir -p ${STUBSPATH}
# Execute command
# exec python ../tracker.py "$@"

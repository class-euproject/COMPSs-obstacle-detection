# COMPSs Obstacle Detection 

This repository contains the COMPSs Obstacle Detection application file and the configuration files required to execute it.


## Prerequisites

The following software components are **required** in order to fully execute the application workflow described in this repository.


### tkDNN

Download and install [object detection (tkDNN)](https://github.com/class-euproject/class-edge/tree/bsc) software, installation and execution instructions are available at the same repository. This component **must** be running at the **edge resource running the COMPSs Obstacle Detection application** when executing it.

<!-- TODO: add deduplicator-->

### COMPSs

Download and install [COMPSs](https://github.com/class-euproject/compss/tree/ppc/ilp-cloudprovider-merge), installation and execution instructions are available at the repository in the link.


### dataClay

Download and install [dataClay](https://github.com/class-euproject/dataclay-class), installation and execution instructions are available at the repository. Refer to the **deployment stage**, as the build stage can be skipped as the dockers are already present at Dockerhub. This component **must** be running at the **Cloud resource** when executing the COMPSs Obstacle Detection application.


### PyWren

Download and install [PyWren](https://github.com/class-euproject/pywren-ibm-cloud.git) at the **Cloud resource**, installation and execution instructions are available at the repository from the link.


### Trajectory Prediction

Download and install [trajectory prediction](https://github.com/class-euproject/trajectory-prediction) at the **Cloud resource**, installation and execution instructions are available at the repository from the link.


### Collision Detection

Download and install [collision detection](https://github.com/class-euproject/collision-detection) at the **Cloud resource**, installation and execution instructions are available at the repository from the link. This component is **not** required in order to execute the COMPSs Obstacle Detection application.


### Python dependencies

The application has been developed and tested with `Python 3.6.8`. The following python packages must be installed as well:

```
pip3 install pymap3d zmq struct requests pygeohash geolib
pip3 install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dataClay==2.6.dev20210422
```


## Installation

### Pull submodules

The tracker\_CLASS submodule is pulled by:

```
git submodule init
git submodule update
```

### Compile tracker\_CLASS

To compile the `tracker_CLASS` component, go to the `./tracker_CLASS/` folder after initializing the submodule and use the following commands:

```
mkdir build
cd build
cmake ../
make
```

The previous steps will generate an executable with a name similar to `track.cpython-36m-aarch64-linux-gnu.so`. Use the following command to move the tracker executable next to the COMPSs Obstacle Detection application:

```
mv track*.so ../../track.so
```


### Update configuration files

The dataClay configuration file `./cfgfiles/client.properties` must be updated. The `HOST=${CLOUD_IP}` variable should be replaced to point to the IP address of your cloud.

Moreover, the COMPSs configuration files `project.xml` and `resources.xml` are used to define the computing resources that will be used to execute the tasks composing this application, which can be found in `./config/`. These files should be updated by replacing the IP address **IP1** for the actual IP address of your computing resource, the username **USER1** and the **PATH1** for the current absolute path to where the `app.py` is located inside your computing resource. For each computing resource that is going to be used, you need to add the `<ComputeNode>` XML tag with the required information described above.


## Execution 

To launch the application use the following script:

```
runcompss --lang=python --project=config/project.xml --resources=config/resources.xml --cpu_affinity=disabled --python_interpreter=python3 --scheduler="es.bsc.compss.scheduler.rtheuristics.RTHeuristicsScheduler" --scheduler_config_file=config/sched.config app.py
```

NOTE: To remind again that the application relies on the **tkDNN** component being up and running on the **edge** resource, and **dataClay** and **Pywren** components being up and running on the **Cloud** resource. Follow the instructions to execute the different components in their respective repositories.

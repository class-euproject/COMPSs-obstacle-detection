# COMPSs Obstacle Detection 

This repository contains the COMPSs Obstacle Detection application file and the configuration files required to execute it.


## Prerequisites

The following software components are **required** in order to fully execute the application workflow described in this repository.


### tkDNN

Download and install [object detection (tkDNN)](https://github.com/class-euproject/class-edge/tree/bsc) software, installation and execution instructions are available at the same repository. This component **must** be running at the **edge resource running the COMPSs Obstacle Detection application** when executing it.


### COMPSs

Download and install [COMPSs](https://github.com/class-euproject/compss/tree/ppc/ilp-cloudprovider-merge), installation and execution instructions are available at the repository in the link.


### dataClay

Download and install [dataClay](https://github.com/class-euproject/dataclay-class), installation and execution instructions are available at the repository. Refer to the **deployment stage**, as the build stage can be skipped as the dockers are already present at Dockerhub. This component **must** be running at the **Cloud resource** when executing the COMPSs Obstacle Detection application.


### PyWren

Download and install [PyWren](https://github.com/class-euproject/pywren-ibm-cloud.git) at the **Cloud resource**, installation and execution instructions are available at the repository from the link.


### Trajectory Prediction

Download and install [trajectory prediction](https://github.com/class-euproject/trajectory-prediction) at the **Cloud resource**, installation and execution instructions are available at the repository from the link.


### Collision Detection

Download and install [collision detection](https://github.com/class-euproject/collision-detection) at the **Cloud resource**, installation and execution instructions are available at the repository from the link.


## Execution

### Pull submodules

The tracker\_CLASS submodule is pulled by:

```
git submodule init
git submodule update
```

### Compile tracker\_CLASS

To compile the `tracker\_CLASS` component, go to the `./tracker\_CLASS/` folder after initializing the submodule and use the following commands:

```
mkdir build
cd build
cmake ../
make
```

The previous steps will generate a `track.cpython-36m-aarch64-linux-gnu.so`. Use the following command to move the tracker executable next to the COMPSs Obstacle Detection application:

```
mv track.cpython-36m-aarch64-linux-gnu.so ../../track.so
```


### Update configuration files

The dataClay configuration file `./cfgfiles/client.properties` must be updated. The `HOST=${IP}` variable should be replaced to point to the IP address of your cloud.

Moreover, the COMPSs configuration files `project.xml` and `resources.xml` can be found in `./config/`. In the example files being provided 3 computing resources are being considered, but these files should be updated by replacing the IP addresses and the usernames by the ones of the computing resources being used.


### Launch the application

To launch the application use the following script:

```
./launch_app.sh
```

NOTE: To remind again that the application relies on the **tkDNN** component being up and running on the **edge** resource, and **dataClay**, **Pywren** and **Trajectory Prediction** components being up and running on the **Cloud** resource.

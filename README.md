# computer_vision for robotX

## preinstall

use pip to install required APIs
```bash
pip install imutils
pip install numpy
pip install opencv-python
```

## Usage

Run vrx_challenges without a dock. For example:

```bash
roslaunch vrx_gazebo navigation_task.launch urdf:=$HOME/ntu_vrx/ntu_wamv/ntu_wamv.urdf
```

then run
```bash
roslaunch computer_vision objectdetection.launch
```


## Return data

publisher: "ntu/img_item_pub",

to see the publisher

```bash
rostopic echo /ntu/img_item_pub
```

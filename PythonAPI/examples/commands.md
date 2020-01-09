# Tab 1
1. `carla_fixed_launch`

# Tab 2
1. `carla_config`
2. `roslaunch carla_ros_bridge carla_ros_bridge.launch`

# Tab 3
1. `cd ~/carla/PythonAPI/examples`
2. `python3 run_test.py -i <ID> -t <TRAIL_NUM> -r <RECORD>`
(Forward - Reverse) * 3
Intructions:
**Goal: To park safely to any free spot given the prescribed maneuver (forward / reverse)** 
1. Acc first then brake at the first epoc of each trail
2. Reset the gear to neutral after finishing parking
3. Log the intention when you decides, before starting maneuver
4. Press the furtherest left red button to start the next epoc
5. 

# Tab 4
1. `mv ~/.config/Epic/CarlaUE4/Saved/parking*.log ~/carla/PythonAPI/examples/bags`
2. Upload the data to Google Drive and move the bags to D:/

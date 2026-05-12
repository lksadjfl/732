import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/afs/ec.auckland.ac.nz/users/j/x/jxia219/unixhome/ros2_ws/install/tb4_sensor_reader'

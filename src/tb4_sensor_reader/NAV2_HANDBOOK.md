# Nav2 测试说明

当前 Nav2、红方块检测和自动返航流程已经合并到统一 Phase 2 入口。

请使用完整手册：

```text
src/tb4_sensor_reader/TEST_HANDBOOK.txt
```

Phase 2 启动命令：

```bash
ros2 launch tb4_sensor_reader phase2_nav2_red_return.launch.py \
  namespace:=/T21 \
  map:=$HOME/ros2_ws/maps/phase1_env_data_map.yaml
```

不要同时启动旧入口 `phase2_with_map.launch.py` 或旧自主节点
`map_frame_avoidance`。

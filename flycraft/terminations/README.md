# Termination document

## Direction description

### .py in root dir

Termination used by velocity vector control.

1. 坠机：通过高度小于0判断(crash_termination.py)
2. 持续远离目标：连续2秒，当前状态和目标状态之间的角度差增大(continuousely_move_away_termination.py)
3. 负过载且大幅度滚转：nz<0且phi>60(deg)连续超过2秒(negtive_overload_and_big_phi_termination.py)
4. 连续滚转：连续滚转2圈
5. 超时：40秒内未达到目标速度矢量

### for_attitude_control

Termination used by attitude control.

### for_BFM_level_turn

Termination used by BFM level turn.

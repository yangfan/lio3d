[pointcloud dataset](https://lgg.epfl.ch/statues_dataset.php)

### dataset

1. ULHK:

- no time info
- time unit: microseconds
- issue: the scan time is estimated by angular velocity which is not accurate enough. Therefore the start time of a scan is tens of (around 20 ) ms later than the end time of previous scan.
- cloud topic name: /velodyne_points_0
- imu topic name: /imu/data

2. NCLT:

- has time info
- time unit: seconds
- cloud topic name: points_raw
- imu topic name: imu_raw

### Issues

1. eskf:
   1. Covariance matrix of motion noise is too small if dt is included, which makes Kalman gain so small that system rely on mainly on IMU prediction.
2. ndt inc:
   1. wrong number of evaluated points, clear points before evaluated: clear and count evaluated points outside of loop.
   2. align optimization: count valid point number in parallel version of for_each function

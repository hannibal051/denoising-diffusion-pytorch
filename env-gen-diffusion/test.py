import numpy as np
import matplotlib.pyplot as plt

# Vehicle parameters
max_steering_angle = np.deg2rad(30)  # Maximum steering angle in radians
vehicle_speed = 1.0  # Constant vehicle speed

# PD controller parameters
Kp = 1.0  # Proportional gain
Kd = 0.6  # Derivative gain

# Define the track as a set of waypoints (x, y coordinates)
track_waypoints = [(0, 0), (1, 1), (2, 0), (3, -1), (4, 0), (5, 1), (6, 0)]

# Initial vehicle state (position, orientation, and error)
x = 0.0
y = 0.0
theta = 0.0
error = 0.0

# Lists to store vehicle states for plotting
x_history = []
y_history = []

# Simulation parameters
dt = 0.1  # Time step
sim_duration = 10.0  # Duration of simulation
error_prev = 0.

for _ in np.arange(0, sim_duration, dt):
    # Calculate the desired heading (angle to the next waypoint)
    desired_heading = np.arctan2(track_waypoints[0][1] - y, track_waypoints[0][0] - x)

    # Calculate the cross-track error (distance from the vehicle to the track)
    error = np.sin(desired_heading - theta) * np.linalg.norm([track_waypoints[0][0] - x, track_waypoints[0][1] - y])

    # Calculate control inputs (steering angle)
    steering_angle = Kp * error + Kd * (error - error_prev)

    # Limit the steering angle to the maximum allowable
    steering_angle = np.clip(steering_angle, -max_steering_angle, max_steering_angle)

    # Update vehicle state
    x += vehicle_speed * np.cos(theta) * dt
    y += vehicle_speed * np.sin(theta) * dt
    theta += (vehicle_speed / 1.0) * np.tan(steering_angle) * dt  # Vehicle model: Ackermann steering
    error_prev = error

    # Store vehicle states for plotting
    x_history.append(x)
    y_history.append(y)

# Plot the vehicle path and track
plt.figure(figsize=(8, 6))
plt.plot(*zip(*track_waypoints), marker='o', linestyle='-', color='r', label='Track')
plt.plot(x_history, y_history, marker='.', linestyle='-', color='b', label='Vehicle Path')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Vehicle Following a Track with PD Control')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from flask import Flask, render_template, jsonify, request
import requests
import geopandas as gpd
import shapely


class OceanRoutingEnv(gym.Env):
    def __init__(self, ports_data, wind_data, current_data):
        super().__init__()
        self.ports = ports_data
        self.wind_data = wind_data
        self.current_data = current_data

        # Refined action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-0.05, high=0.05, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [-90, -180, -90, -180, 0, 0, 0, 0]
            ),  # Lat: -90 to 90, Lon: -180 to 180
            high=np.array(
                [90, 180, 90, 180, 100, 360, 100, 360]
            ),  # Speed and direction bounds
            dtype=np.float32,
        )

        self.current_pos = None
        self.target_pos = None
        self.max_steps = 500
        self.steps = 0
        self.path_history = []
        self.land_polygons = self._load_land_polygons()

    def _load_land_polygons(self):
        """Load land polygons from Natural Earth dataset"""
        try:
            # Load shapefile
            gdf = gpd.read_file("ne_50m_land/ne_50m_land.shp")
            self.land_geom = gdf.geometry
            return gdf

        except Exception as e:
            print(f"Warning: Failed to load land polygons: {e}")
            self.land_geom = None
            self.land_index = None
            return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "coordinates" in options:
            start_coords, end_coords = options["coordinates"]
            self.current_pos = np.array(start_coords, dtype=np.float32)
            self.target_pos = np.array(end_coords, dtype=np.float32)
        else:
            # Use default coordinates from ports_data if available
            self.current_pos = np.array(self.ports["start"], dtype=np.float32)
            self.target_pos = np.array(self.ports["end"], dtype=np.float32)

        self.path_history = [self.current_pos.tolist()]
        self.steps = 0

        return self._get_observation(), {}

    def step(self, action):
        # Store previous position
        prev_pos = self.current_pos.copy()

        # Calculate movement
        wind_effect = self._calculate_wind_effect()
        current_effect = self._calculate_current_effect()
        total_movement = action + wind_effect + current_effect

        # Update position
        self.current_pos += total_movement

        # Force stay near start point in first few steps
        if self.steps < 5:
            start_dist = self._haversine_distance(
                self.current_pos, np.array(self.ports["start"])
            )
            if start_dist > 1.0:
                self.current_pos = prev_pos  # Revert movement

        # Handle final approach
        dist_to_target = self._haversine_distance(self.current_pos, self.target_pos)
        if dist_to_target < 2.0:  # Within final approach
            # Pull towards exact target position
            direction = self.target_pos - self.current_pos
            self.current_pos += direction * 0.1

        self.path_history.append(self.current_pos.tolist())
        self.steps += 1

        # Rest of the existing step logic
        distance = self._haversine_distance(self.current_pos, self.target_pos)
        if self._is_over_land():
            return (
                self._get_observation(),
                -5000000,
                True,
                False,
                {"termination": "land_collision"},
            )

        done = distance < 0.1 or self.steps >= self.max_steps
        reward = self._calculate_reward(distance, done)

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        return np.array(
            [
                self.current_pos[0],  # Current Latitude
                self.current_pos[1],  # Current Longitude
                self.target_pos[0],  # Target Latitude
                self.target_pos[1],  # Target Longitude
                self.wind_data["speed"],
                self.wind_data["direction"],
                self.current_data["speed"],
                self.current_data["direction"],
            ]
        )

    def _calculate_wind_effect(self):
        speed = self.wind_data["speed"]
        direction = np.radians(self.wind_data["direction"])
        return np.array(
            [speed * np.cos(direction) / 6371, speed * np.sin(direction) / 6371]
        )

    def _calculate_current_effect(self):
        speed = self.current_data["speed"]
        direction = np.radians(self.current_data["direction"])
        return np.array(
            [speed * np.cos(direction) / 6371, speed * np.sin(direction) / 6371]
        )

    def _is_over_land(self):
        """Check if current position intersects with land"""
        if self.land_geom is None:
            print(f"Warning: Land loading failed: {e}")  # noqa: F821
            return True

        try:
            # Create point from current position
            point = shapely.geometry.Point(self.current_pos[0], self.current_pos[1])

            if self.land_geom.contains(point).any():
                print(
                    f"Land Hit Status at Longitude: {self.current_pos[0]} Latitude: {self.current_pos[1]}: {self.land_geom.contains(point).any()}"
                )

            return self.land_geom.contains(point).any()

        except Exception as e:
            print(f"Warning: Land detection failed: {e}")
            return True

        except Exception as e:
            print(f"Warning: Land intersection check failed: {e}")
            return True

    def _haversine_distance(self, pos1, pos2):
        R = 6371  # Earth radius in km
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def _calculate_reward(self, distance, done):
        # Base rewards
        efficiency_bonus = 10 / (1 + distance)
        distance_penalty = -distance * 2

        # Start/End point penalties
        start_dist = self._haversine_distance(
            self.current_pos, np.array(self.ports["start"])
        )
        if (
            self.steps < 10 and start_dist > 0.5
        ):  # Penalize if moving away from start too early
            return -10000

        # Completion rewards/penalties
        if done:
            end_dist = self._haversine_distance(self.current_pos, self.target_pos)
            if end_dist < 0.1:  # Successfully reached destination
                return 200000
            else:  # Failed to reach destination
                return -50000

        return distance_penalty + efficiency_bonus


def get_environmental_data(lat, lon):
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m"
        )
        data = response.json()

        # Get first hour's wind data
        wind_speed = data["hourly"]["wind_speed_10m"][0]
        wind_direction = data["hourly"]["wind_direction_10m"][0]

        print(
            f"Wind Speed = {wind_speed}, Wind Direction = {wind_direction} at {lat} {lon}"
        )
        # Approximate ocean currents as 30% of wind speed, same direction
        current_speed = wind_speed * 0.3
        current_direction = wind_direction

        return {
            "wind": {"speed": wind_speed, "direction": wind_direction},
            "current": {"speed": current_speed, "direction": current_direction},
        }

    except Exception as e:
        print(f"API Error: {e}")
        return {
            "wind": {"speed": 10, "direction": 45},
            "current": {"speed": 3, "direction": 30},
        }


app = Flask(__name__, static_folder="static", template_folder="templates")


PORT_COORDINATES = {
    "Mumbai": [72.8463, 18.9335],
    "Chennai": [80.3416, 13.0937],
    "Kolkata": [88.1232, 21.1024],
    "Cochin": [76.0673, 9.8937],
    "Visakhapatnam": [83.3916, 17.6882],
    "Colombo": [79.8215, 6.9259],
    "Trincomalee": [81.2287, 8.5936],
    "Chittagong": [91.8357, 22.3193],  # Added Chittagong
}


def get_port_coordinates(port_name):
    return PORT_COORDINATES.get(port_name)


@app.route("/")
def index():
    return render_template("index.html", ports=PORT_COORDINATES)


@app.route("/optimize_route", methods=["POST"])
def optimize_route():
    try:
        start_port = request.json.get("start_port")
        end_port = request.json.get("end_port")

        if not start_port or not end_port:
            return jsonify({"error": "Start and end ports required"}), 400

        # Get port coordinates
        start_coords = get_port_coordinates(start_port)
        end_coords = get_port_coordinates(end_port)

        if not start_coords or not end_coords:
            return jsonify({"error": "Invalid port names"}), 400

        # Get environmental data
        env_data = get_environmental_data(
            (start_coords[0] + end_coords[0]) / 2, (start_coords[1] + end_coords[1]) / 2
        )

        # Create environment with proper initialization
        env = OceanRoutingEnv(
            ports_data={"start": start_coords, "end": end_coords},
            wind_data=env_data["wind"],
            current_data=env_data["current"],
        )

        # Wrap environment for stable-baselines
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import get_device

        env = Monitor(env)

        # Specify the device (GPU if available)
        device = get_device("auto")  # Automatically selects GPU if available

        # Initialize and train model with proper reset
        model = PPO("MlpPolicy", env, verbose=1, device=device)

        # Initial reset with coordinates
        env.reset(options={"coordinates": (start_coords, end_coords)})

        # Train model
        model.learn(total_timesteps=50000)  # Increased from 10000

        # Generate route
        obs = env.reset(options={"coordinates": (start_coords, end_coords)})[0]
        route = [start_coords]  # Explicitly start with source port
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _, _ = env.step(action)
            current_pos = obs[:2].tolist()
            route.append(current_pos)

        route.append(end_coords)  # Explicitly end with destination port

        return jsonify(
            {
                "route": route,
                "start_port": start_port,
                "end_port": end_port,
                "wind_data": env_data["wind"],
            }
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Route optimization failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)

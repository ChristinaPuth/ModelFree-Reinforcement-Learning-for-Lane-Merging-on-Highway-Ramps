from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle

class MergeEnv1(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
     
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.2,
            "lane_change_reward": 0.1,
        })
        return cfg

    def _reward(self, action: int) -> float:
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        return utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1),
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)
            )
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()
        ends = [150, 80, 80, 150]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self, num_CAV=1, num_HDV=8) -> None:
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        spawn_points_s = [10,25,33,40,50,55,65,75,80, 90, 130, 170, 210, 230, 250]
        spawn_points_m = [55]

        np.random.shuffle(spawn_points_s)
        np.random.shuffle(spawn_points_m)

        max_s = len(spawn_points_s)
        max_m = len(spawn_points_m)
        total_spawn = max_s + max_m
        if num_CAV + num_HDV > total_spawn:
            raise ValueError(f"Too many vehicles requested ({num_CAV + num_HDV}), only {total_spawn} unique spawn points available.")

        spawn_point_s_c = spawn_points_s[:min(num_CAV // 2, max_s)]
        spawn_point_m_c = spawn_points_m[:num_CAV - len(spawn_point_s_c)]
        remaining_s = spawn_points_s[len(spawn_point_s_c):]
        remaining_m = spawn_points_m[len(spawn_point_m_c):]
        spawn_point_s_h = remaining_s[:min(num_HDV // 2, len(remaining_s))]
        spawn_point_m_h = remaining_m[:num_HDV - len(spawn_point_s_h)]

        total_vehicles = len(spawn_point_s_c) + len(spawn_point_m_c) + len(spawn_point_s_h) + len(spawn_point_m_h)
        initial_speeds = np.random.rand(total_vehicles) * 2 + 25
        loc_noise = np.random.rand(total_vehicles) * 3 - 1.5

        idx = 0

        # CAVs on main and merging road
        for p in spawn_point_s_c + spawn_point_m_c:
            lane = ("a", "b", 0) if p in spawn_point_s_c else ("j", "k", 0)
            pos = road.network.get_lane(lane).position(p + loc_noise[idx], 0)
            speed = initial_speeds[idx]
            ego = self.action_type.vehicle_class(road, pos, speed)
            self.controlled_vehicles.append(ego)
            road.vehicles.append(ego)
            idx += 1

        # HDVs on main and merging road
        for p in spawn_point_s_h + spawn_point_m_h:
            lane = ("a", "b", 0) if p in spawn_point_s_h else ("j", "k", 0)
            pos = road.network.get_lane(lane).position(p + loc_noise[idx], 0)
            speed = initial_speeds[idx]
            vehicle = ControlledVehicle(road, pos, speed=speed)  # ✅ Ensure it's a ControlledVehicle
            vehicle.target_speed = 25                             # ✅ Set other car's speed
            road.vehicles.append(vehicle)
            idx += 1

        # Set the ego vehicle
        if self.controlled_vehicles:
            self.vehicle = self.controlled_vehicles[0]


    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


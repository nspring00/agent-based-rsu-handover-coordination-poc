import math
import unittest

from mesa import Model

from poc.strategies import StaticVehicleLoadGenerator, ARHCStrategy, is_moving_towards
from poc.model import VehicleAgent, VECStationAgent, VECModel


class TestIsMovingTowards(unittest.TestCase):
    def test_moving_directly_towards(self):
        self.assertTrue(is_moving_towards((0, 0), 45, (1, 1)), "Vehicle should be moving towards the station")
        self.assertTrue(is_moving_towards((0, 0), 0, (10, 0)),
                        "Vehicle should be moving directly towards the station on the x-axis")
        self.assertTrue(is_moving_towards((5, 5), 45, (10, 10)),
                        "Vehicle should be moving towards the station in a diagonal direction")
        self.assertTrue(is_moving_towards((10, 10), 180, (0, 10)),
                        "Vehicle should be moving towards the station on the x-axis")
        self.assertTrue(is_moving_towards((-10, 10), 270, (-5, -20)), "Vehicle should be moving towards the station")

    def test_moving_away(self):
        self.assertFalse(is_moving_towards((0, 0), 225, (1, 1)), "Vehicle should be moving away from the station")

    def test_moving_orthogonal(self):
        self.assertFalse(is_moving_towards((0, 0), 136, (1, 1)), "Vehicle should be moving orthogonal to the station")
        self.assertFalse(is_moving_towards((0, 0), 314, (1, 1)), "Vehicle should be moving orthogonal to the station")

    def test_moving_directly_away(self):
        self.assertFalse(is_moving_towards((10, 0), 180, (20, 0)),
                         "Vehicle should be moving directly away from the station on the x-axis")


class TestVECStationAgent(unittest.TestCase):
    def test_calculate_station_bearing(self):
        class MockModel(Model):
            def __init__(self):
                super().__init__()

        # noinspection PyTypeChecker
        model: VECModel = MockModel()
        station = VECStationAgent(0, model, ARHCStrategy(), (10, 10), 10, 10)

        vehicle = VehicleAgent(0, model, None, StaticVehicleLoadGenerator(), 1)

        vehicle.pos = (15, 10)
        self.assertAlmostEqual(station.calculate_vehicle_station_bearing(vehicle), 0)
        vehicle.pos = (15, 15)
        self.assertAlmostEqual(station.calculate_vehicle_station_bearing(vehicle), math.pi / 4)
        vehicle.pos = (10, 15)
        self.assertAlmostEqual(station.calculate_vehicle_station_bearing(vehicle), math.pi / 2)
        vehicle.pos = (5, 15)
        self.assertAlmostEqual(station.calculate_vehicle_station_bearing(vehicle), 3 * math.pi / 4)
        vehicle.pos = (5, 10)
        self.assertAlmostEqual(station.calculate_vehicle_station_bearing(vehicle), math.pi)
        vehicle.pos = (5, 5)
        self.assertAlmostEqual(station.calculate_vehicle_station_bearing(vehicle), -3 * math.pi / 4)
        vehicle.pos = (10, 5)
        self.assertAlmostEqual(station.calculate_vehicle_station_bearing(vehicle), -math.pi / 2)
        vehicle.pos = (15, 5)
        self.assertAlmostEqual(station.calculate_vehicle_station_bearing(vehicle), -math.pi / 4)

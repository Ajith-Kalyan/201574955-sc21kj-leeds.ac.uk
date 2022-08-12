import glob
import os
import random
import sys
import time

import gym
import pygame
import numpy as np
import cv2

from gym import spaces

from carla_sync_mode import CarlaSyncMode
from carla_weather import Weather
from agents.navigation.roaming_agent import RoamingAgent

#Locate Carla ".egg file"
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ObstacleDetectionEvent as od


class CarlaEnv(gym.Env):

    def __init__(self,
                 render,
                 carla_port,
                 changing_weather_speed,
                 frame_skip,
                 observations_type,
                 traffic,
                 vehicle_name,
                 map_name,
                 autopilot):

        super(CarlaEnv, self).__init__()
        self.render_display = render
        self.changing_weather_speed = float(changing_weather_speed)
        self.frame_skip = frame_skip
        self.observations_type = observations_type
        self.traffic = traffic
        self.vehicle_name = vehicle_name
        self.map_name = map_name
        self.autopilot = autopilot
        self.actor_list = []
        self.lane_history = []
        self.distance = 0.0
        self.count = 0

        # initialize rendering
        if self.render_display:
            pygame.init()
            self.render_display = pygame.display.set_mode(
                (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()

        # initialize client with timeout
        self.client = carla.Client('localhost', carla_port)
        self.client.set_timeout(10.0)

        # initialize world and map
        self.world = self.client.load_world(self.map_name)
        self.map = self.world.get_map()

        # remove old vehicles and sensors (in case they survived)
        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter('*vehicle*'):
            print('Warning: removing old vehicle')
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print('Warning: removing old sensor')
            sensor.destroy()

        # create vehicle
        self.vehicle = None
        self.vehicles_list = []
        self._reset_vehicle()
        self.actor_list.append(self.vehicle)
        
        # initialize blueprint library
        blueprint_library = self.world.get_blueprint_library()

        # spawn camera for rendering
        if self.render_display:
            self.camera_display = self.world.spawn_actor(blueprint_library.find('sensor.camera.rgb'),
                                                         carla.Transform(carla.Location(x=2.5, z=0.7)), attach_to=self.vehicle)
            self.actor_list.append(self.camera_display)

        # spawn camera for pixel observations
        if self.observations_type == 'pixel':
            bp = blueprint_library.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(168))
            bp.set_attribute('image_size_y', str(168))
            bp.set_attribute('fov', str(100))
            location = carla.Location(x=1.6, z=1.7)
            self.camera_vision = self.world.spawn_actor(
                bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
            self.actor_list.append(self.camera_vision)

        # context manager initialization
        if self.render_display and self.observations_type == 'pixel':
            self.sync_mode = CarlaSyncMode(
                self.world, self.camera_display, self.camera_vision, fps=20)
        elif self.render_display and self.observations_type == 'state':
            self.sync_mode = CarlaSyncMode(
                self.world, self.camera_display, fps=20)
        elif not self.render_display and self.observations_type == 'pixel':
            self.sync_mode = CarlaSyncMode(
                self.world, self.camera_vision, fps=20)
        elif not self.render_display and self.observations_type == 'state':
            self.sync_mode = CarlaSyncMode(self.world, fps=20)
        else:
            raise ValueError(
                'Unknown observation_type. Choose between: state, pixel')

        # weather
        self.weather = Weather(self.world, self.changing_weather_speed)

        # collision detection
        self.collision = False
        sensor_blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            sensor_blueprint, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Adding lane detection
        lane_sensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_sensor = self.world.spawn_actor(
            lane_sensor, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(lambda event: self._on_lane_change(event))

        # Obstacle detection
        obstacle_sensor = self.world.get_blueprint_library().find('sensor.other.radar')
        self.obstacle_sensor = self.world.spawn_actor(
            sensor_blueprint, carla.Transform(), attach_to=self.vehicle)
        self.obstacle_sensor.listen(
            lambda event: self._on_obstacle_detected(event))

        # initialize autopilot
        self.agent = RoamingAgent(self.vehicle)

        # get initial observatio
        if self.observations_type == 'state':
            obs = self._get_state_obs()
        else:
            obs = np.zeros((168, 168))

        # gym environment specific variables
        #Observaton space and action space
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        self.obs_dim = obs.shape
        self.observation_space = spaces.Box(-np.inf,
                                            np.inf, shape=self.obs_dim, dtype='float32')
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))

    def reset(self):
        self._reset_vehicle()
        self.world.tick()
        self._reset_other_vehicles()
        self.world.tick()
        self.count = 0
        self.collision = False
        self.lane_history = []
        self.distance = 0.0
        time.sleep(1.0)
        obs, _, _, _ = self.step([0, 0])
        return obs

    def _reset_vehicle(self):
        # choose random spawn point
        init_transforms = self.world.get_map().get_spawn_points()
        vehicle_init_transform = random.choice(init_transforms)

        # create the vehicle
        if self.vehicle is None:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find(
                'vehicle.' + self.vehicle_name)
            if vehicle_blueprint.has_attribute('color'):
                color = random.choice(
                    vehicle_blueprint.get_attribute('color').recommended_values)
                vehicle_blueprint.set_attribute('color', color)
            self.vehicle = self.world.spawn_actor(
                vehicle_blueprint, vehicle_init_transform)
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=0.0))
            time.sleep(0.2)
        else:
            self.vehicle.set_transform(vehicle_init_transform)

    def _reset_other_vehicles(self):
        if not self.traffic:
            return

        # clear out old vehicles
        self.client.apply_batch([carla.command.DestroyActor(x)
                                for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []

        # initialize traffic manager
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.global_percentage_speed_difference(30.0)
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(
            x.get_attribute('number_of_wheels')) == 4]

        # choose random spawn points
        num_vehicles = 150
        init_transforms = self.world.get_map().get_spawn_points()
        init_transforms = np.random.choice(init_transforms, num_vehicles)

        # spawn vehicles
        batch = []
        for transform in init_transforms:
            transform.location.z += 0.1  # otherwise can collide with the road it starts on
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(
                    blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True)))

        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)

    def _compute_action(self):
        return self.agent.run_step()

    def step(self, action):
        rewards = []
        next_obs, done, info = np.array([]), False, {}
        for _ in range(self.frame_skip):
            if self.autopilot:
                vehicle_control = self._compute_action()
                steer = float(vehicle_control.steer)
                if vehicle_control.throttle > 0.0 and vehicle_control.brake == 0.0:
                    throttle_brake = vehicle_control.throttle
                elif vehicle_control.brake > 0.0 and vehicle_control.throttle == 0.0:
                    throttle_brake = - vehicle_control.brake  
                else:
                    throttle_brake = 0.0
                action = [throttle_brake, steer]
            next_obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)
            if done:
                break
        return next_obs, np.mean(rewards), done, info

    def _simulator_step(self, action):
        if self.render_display:
            if should_quit():
                return
            self.clock.tick()

        # calculate actions
        throttle_brake = 0.5+float(action[0])
        steer = float(action[1])
        if throttle_brake >= 0.0:
            throttle = throttle_brake
            if throttle > 12:
                throttle = 12
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        # apply control to simulation
        vehicle_control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )

        self.vehicle.apply_control(vehicle_control)

        # advance the simulation and wait for the data
        if self.render_display and self.observations_type == 'pixel':
            snapshot, display_image, vision_image = self.sync_mode.tick(
                timeout=2.0)
        elif self.render_display and self.observations_type == 'state':
            snapshot, display_image = self.sync_mode.tick(timeout=2.0)
        elif not self.render_display and self.observations_type == 'pixel':
            snapshot, vision_image = self.sync_mode.tick(timeout=2.0)
        elif not self.render_display and self.observations_type == 'state':
            self.sync_mode.tick(timeout=2.0)
        else:
            raise ValueError(
                'Unknown observation_type. Choose between: state, pixel')

        # Weather evolves
        self.weather.tick()
        acceleration = vector_to_scalar(self.vehicle.get_acceleration())
        # draw the display
        if self.render_display:
            draw_image(self.render_display, display_image)
            self.render_display.blit(self.font.render(
                'Frame: %d' % self.count, True, (255, 255, 255)), (8, 10))
            self.render_display.blit(self.font.render(
                'Thottle: %f' % acceleration, True, (255, 255, 255)), (8, 28))
            self.render_display.blit(self.font.render(
                'Steer: %f' % steer, True, (255, 255, 255)), (8, 46))
            self.render_display.blit(self.font.render(
                'Brake: %f' % brake, True, (255, 255, 255)), (8, 64))
            self.render_display.blit(self.font.render(
                str(self.weather), True, (255, 255, 255)), (8, 82))
            pygame.display.flip()

        # get reward and next observation
        reward, done, info = self._get_reward()
        if self.observations_type == 'state':
            next_obs = self._get_state_obs()
        else:
            next_obs = self._get_pixel_obs(vision_image)

        # increase frame count
        self.count += 1

        return next_obs, reward, done, info

    def _get_pixel_obs(self, vision_image):

        def roi(img, vertices):
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, vertices, 255)
            masked_image = cv2.bitwise_and(img, mask)
            return masked_image

        array = np.frombuffer(vision_image.raw_data, dtype=np.dtype('uint8'))
        array_1d = np.copy(array)
        array_4d = array_1d.reshape(
            (vision_image.height, vision_image.width, 4))

        h = array_4d.shape[0]
        w = array_4d.shape[1]

        roi_vertices = [(0, h), (3.5*w/10, 5*h/10), (7*w/10, 5*h/10), (w, h)]

        gray = cv2.cvtColor(array_4d, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 5, 60)
        cropped = roi(canny, np.array([roi_vertices], np.int32))

        cv2.imshow("observation", cropped)
        cv2.waitKey(1)
        return cropped

    def _get_state_obs(self):
        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        x_pos = location.x
        y_pos = location.y
        z_pos = location.z
        pitch = rotation.pitch
        yaw = rotation.yaw
        roll = rotation.roll
        acceleration = vector_to_scalar(self.vehicle.get_acceleration())
        angular_velocity = vector_to_scalar(
            self.vehicle.get_angular_velocity())
        velocity = vector_to_scalar(self.vehicle.get_velocity())
        return np.array([x_pos,
                         y_pos,
                         z_pos,
                         pitch,
                         yaw,
                         roll,
                         acceleration,
                         angular_velocity,
                         velocity], dtype=np.float64)

    def _get_reward(self):
        vehicle_location = self.vehicle.get_location()
        follow_waypoint_reward = 0
        is_lane_changed = self._get_lanechange_reward()
        done, collision_reward = self._get_collision_reward()
        acceleration = vector_to_scalar(self.vehicle.get_acceleration())
        distance_to_obs = self._get_distance_to_obstacle()
        cost = self._get_cost()
        acceleration_reward = 0
        time_alive = 0
        is_road = self._get_follow_waypoint_reward(vehicle_location)
        wp_dist, is_road = self._get_follow_waypoint_reward(vehicle_location)

        time_alive += 0.02
        time_alive += 0.001

        wp_dist = round((10*wp_dist), 2)
        waypoint = is_road.lane_type

        # Reward calculation for staying at the centre of lane
        if -3.0 <= wp_dist and acceleration >= 2:
            dist_wp_reward = 5
        elif -0.80 <= wp_dist and acceleration < 2:
            dist_wp_reward = 0
        else:
            dist_wp_reward = (wp_dist)/2

        # Reward Calculation for steering out of road
        if is_road.lane_type == carla.LaneType.Shoulder:
            follow_waypoint_reward = -40
            done = True
        else:
            if is_road.lane_type == carla.LaneType.Driving:
                follow_waypoint_reward = 1

        #Reward Calculation for collision
        collision_reward = round((10*collision_reward), 2)

        # Reward calculation for acceleration
        if acceleration > 10:
            if acceleration > 15:
                acceleration_reward = -5
            elif acceleration < 2 and collision_reward == 1:
                collision_reward = 0


        total_reward = collision_reward + acceleration_reward + is_lane_changed + follow_waypoint_reward + dist_wp_reward
       
        total_reward = round(total_reward, 2)
        if acceleration > 9.7 and acceleration < 9.9:
            total_reward = 0

        self.lane_history = []

        info_dict = dict()
        info_dict['acc'] = acceleration
        info_dict['waypoint'] = follow_waypoint_reward
        info_dict['collision'] = collision_reward
        info_dict['lane'] = is_lane_changed
        info_dict['acc_rew'] = acceleration_reward
        info_dict['dist'] = dist_wp_reward
        info_dict['total_reward'] = total_reward
        print(info_dict)
        return total_reward, done, info_dict

    def _get_follow_waypoint_reward(self, location):
        
        nearest_wp = self.map.get_waypoint(location, project_to_road=True, 
            
            lane_type=(carla.LaneType.Driving | carla.LaneType.Bidirectional |
             carla.LaneType.Shoulder | carla.LaneType.Sidewalk | carla.LaneType.Stop | carla.LaneType.Biking | carla.LaneType.Border | carla.LaneType.Parking | carla.LaneType.Bidirectional
             | carla.LaneType.Median | carla.LaneType.OnRamp | carla.LaneType.OffRamp | carla.LaneType.Any ))
        
        dist_wp = self.map.get_waypoint(location, project_to_road=True, lane_type=(carla.LaneType.Driving ))
        
        distance = np.sqrt(
            (location.x - dist_wp.transform.location.x) ** 2 +
            (location.y - dist_wp.transform.location.y) ** 2
        )
        return -distance,nearest_wp

    def _get_collision_reward(self):
        if not self.collision:
            return False, 0.1 
        else:
            return True, -4 

    def _get_lanechange_reward(self):
        if len(self.lane_history) != 0:
            return -10
        else:
            return 0

    def _on_lane_change(self, event):
        self.lane_history.append(event)

    def _on_obstacle_detected(self, event):
        self.distance = self.obstacle_sensor.raw_data
        print(self.distance)
        pass

    def _get_distance_to_obstacle(self):
        if self.distance < 5:
            return -(int(10*self.distance))
        else:
            return 0

    def _get_cost(self):
        return 0
        #define cost function

    def _on_collision(self, event):
        other_actor = get_actor_name(event.other_actor)
        self.collision = True
        self._reset_vehicle()

    def close(self):
        for actor in self.actor_list:
            actor.destroy()
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x)
                                for x in self.vehicles_list])
        time.sleep(0.5)
        pygame.quit()

    def render(self, mode):
        pass


def vector_to_scalar(vector):
    scalar = np.around(np.sqrt(vector.x ** 2 +
                               vector.y ** 2 +
                               vector.z ** 2), 2)
    return scalar


def draw_image(surface, image, blend=False):
    def roi(img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))

    array_1d = np.copy(array)
    array_4d = array_1d.reshape((image.height, image.width, 4))

    h = array_4d.shape[0]
    w = array_4d.shape[1]
    roi_vertices = [(0, h), (3.5*w/10, 5*h/10), (7*w/10, 5*h/10), (w, h)]

    gray = cv2.cvtColor(array_4d, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 70)
    cropped = roi(canny, np.array([roi_vertices], np.int32))

    # cv2.imshow("result", cropped)
    # cv2.waitKey(1)

    array = array_4d[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(0)
    surface.blit(image_surface, (0, 0))

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def get_actor_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

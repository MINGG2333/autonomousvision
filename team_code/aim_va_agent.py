import os
import json
import datetime
import pathlib
import time
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner
from aim_va.model import MultiTaskImageNetwork
from aim_va.class_converter import sub_classes
from aim_va.data import seg_to_one_hot
from mmseg.apis import inference_segmentor, init_segmentor

import cv2

SAVE_PATH = os.environ.get('SAVE_PATH', None)

# jxy: addition; (add display.py and fix RoutePlanner.py)
from team_code.display import HAS_DISPLAY, Saver, debug_display
from carla_project.src.common import CONVERTER, COLOR
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights


def get_entry_point():
	return 'WaypointSegmentationAgent'


def crop_image(image, crop_width_factor):
	"""
	crop a PIL image, returning a channels-first numpy array.
	"""
	image = Image.fromarray(image)
	(width, height) = (image.width, image.height)
	image = np.asarray(image).copy()
	crop = int(crop_width_factor * width)
	start_x = height//2 - crop//2
	start_y = width//2 - crop//2
	cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
	return cropped_image[:,:]


def scale_image(image, scale):
	"""
	Scale an image
	"""
	image = Image.fromarray(image)
	(width, height) = (image.width // scale, image.height // scale)
	im_resized = image.resize((width, height), resample=Image.NEAREST)
	image = np.asarray(im_resized).copy()
	return image


class WaypointSegmentationAgent(autonomous_agent.AutonomousAgent):
	
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		return AgentSaver

		# jxy: add return AgentSaver and init_ads (setup keep 5 lines); rm save_path;
	def init_ads(self, path_to_conf_file):

		args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
		self.args = json.load(args_file)

		args_file.close()
		self.input_buffer = {'seg_center': deque()}
		

		self.converter = sub_classes[self.args['classes']]

		num_segmentation_classes = len(np.unique(self.converter)) 

		self.net = MultiTaskImageNetwork('cuda', num_segmentation_classes, self.args['pred_len'], 1)
		
		self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_model.pth')))
		self.net.cuda()
		self.net.eval()

		seg_checkpoint_path = os.path.join(path_to_conf_file, 'iter_80000.pth')
		seg_conf_path = os.path.join(path_to_conf_file, 'fcn_r50-d8_512x1024_80k_carla_full.py')
		self.segmentation_net = init_segmentor(seg_conf_path, seg_checkpoint_path)

		# self.save_path = None
		# if SAVE_PATH is not None:
		# 	now = datetime.datetime.now()
		# 	string = pathlib.Path(os.environ['ROUTES']).stem + '_'
		# 	string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

		# 	self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
		# 	self.save_path.mkdir(parents=True, exist_ok=False)

		# 	(self.save_path / 'seg_center').mkdir()

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

		super()._init() # jxy add

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 800, 'height': 600, 'fov': 100,
					'id': 'rgb_center'
					},
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					},
				# jxy: addition
				{
					'type': 'sensor.camera.semantic_segmentation',
					'x': 0.0, 'y': 0.0, 'z': 100.0,
					'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
					'width': 512, 'height': 512, 'fov': 5 * 10.0,
					'id': 'map'
					},
				]

	def tick(self, input_data):
		self.step += 1
		_rgb = crop_image(input_data['rgb_center'][1][:, :, :3], self.args['input_crop']) # from 800 * 600 to 512 * 512

		result = inference_segmentor(self.segmentation_net, _rgb) 

		seg = result[0].astype(np.uint8)

		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		self.speed = speed

		result = {
				'seg_center': scale_image(seg, 2), # from 512 * 512 to 256 * 256
				'gps': gps,
				'speed': speed,
				'compass': compass,
				}
		
		pos = self._get_position(result)
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value

		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		# jxy addition:
		result['rgb'] = cv2.cvtColor(input_data['rgb_center'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		result['gps'] = pos
		result['far_command'] = next_cmd

		result['R_pos_from_head'] = R
		result['offset_pos'] = np.array([pos[0], pos[1]])

		self._actors = self._world.get_actors()
		self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))
		topdown = input_data['map'][1][:, :, 2]
		topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)
		result['topdown'] = COLOR[CONVERTER[topdown]]
		return result

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		
		tick_data = self.tick(input_data)

		if self.step < self.args['seq_len']:
			seg_center = seg_to_one_hot(np.array((Image.fromarray(tick_data['seg_center'], 'L'))), self.converter).unsqueeze(0)
			self.input_buffer['seg_center'].append(seg_center.to('cuda', dtype=torch.float32))

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			self.record_step(tick_data, control) # jxy: add
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
											torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		encoding = []
		seg_center = seg_to_one_hot(np.array((Image.fromarray(tick_data['seg_center']))), self.converter).unsqueeze(0)
		self.input_buffer['seg_center'].popleft()
		self.input_buffer['seg_center'].append(seg_center.to('cuda', dtype=torch.float32))
		encoding.append(self.net.image_encoder(list(self.input_buffer['seg_center'])))
		
		pred_wp = self.net(encoding, target_point)
		
		 # jxy: points_world
		steer, throttle, brake, metadata, points_world = self.net.control_pid(pred_wp, gt_velocity)
		self.pid_metadata = metadata

		if brake < 0.05: brake = 0.0
		if throttle > brake: brake = 0.0

		control = carla.VehicleControl()
		control.steer = float(steer)
		control.throttle = float(throttle)
		control.brake = float(brake)

		if HAS_DISPLAY: # jxy: change
			debug_display(tick_data, control.steer, control.throttle, control.brake, self.step)

		self.record_step(tick_data, control, points_world) # jxy: add
		return control

	# jxy: add record_step
	def record_step(self, tick_data, control, pred_waypoint=[]):
		# draw pred_waypoint
		if len(pred_waypoint):
			pred_waypoint[:,1] *= -1
			pred_waypoint = tick_data['R_pos_from_head'].dot(pred_waypoint.T).T
		self._route_planner.run_step2(pred_waypoint, is_gps=False, store=False) # metadata['wp_1'] relative to ego head (as y)
		# addition: from leaderboard/team_code/auto_pilot.py
		speed = tick_data['speed']
		self._recorder_tick(control) # trjs
		ego_bbox = self.gather_info() # metrics
		self._route_planner.run_step2(ego_bbox + tick_data['offset_pos'], is_gps=True, store=False)
		self._route_planner.show_route()

		if self.save_path is not None and self.step % self.record_every_n_step == 0:
			self.save(control.steer, control.throttle, control.brake, tick_data)


# jxy: mv save in AgentSaver & rm destroy
class AgentSaver(Saver):
	def __init__(self, path_to_conf_file, dict_, list_):
		self.config_path = path_to_conf_file

		# jxy: according to sensor
		self.rgb_list = ['rgb', 'seg_center', 'topdown', ] # 'bev', 'rgb_left', 'rgb_right', 'rgb_rear'
		self.add_img = [] # 'flow', 'out', 
		self.lidar_list = [] # 'lidar_0', 'lidar_1',
		self.dir_names = self.rgb_list + self.add_img + self.lidar_list + ['pid_metadata']

		super().__init__(dict_, list_)

	def run(self): # jxy: according to init_ads
		path_to_conf_file = self.config_path
		args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
		self.args = json.load(args_file)

		args_file.close()

		super().run()

	def _save(self, tick_data):	
		# addition
		# save_action_based_measurements = tick_data['save_action_based_measurements']
		self.save_path = tick_data['save_path']
		if not (self.save_path / 'ADS_log.csv' ).exists():
			# addition: generate dir for every total_i
			self.save_path.mkdir(parents=True, exist_ok=True)
			for dir_name in self.dir_names:
				(self.save_path / dir_name).mkdir(parents=True, exist_ok=False)

			# according to self.save data_row_list
			title_row = ','.join(
				['frame_id', 'far_command', 'speed', 'steering', 'throttle', 'brake',] + \
				self.dir_names
			)
			with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
				f_out.write(title_row+'\n')

		self.step = tick_data['frame']
		self.save(tick_data['steer'],tick_data['throttle'],tick_data['brake'], tick_data)

	# addition: modified from leaderboard/team_code/auto_pilot.py
	def save(self, steer, throttle, brake, tick_data):
		# frame = self.step // 10
		frame = self.step

		# 'gps' 'thetas'
		pos = tick_data['gps']
		speed = tick_data['speed']
		far_command = tick_data['far_command']
		data_row_list = [frame, far_command.name, speed, steer, throttle, brake,]

		if frame >= self.args['seq_len']: # jxy: according to run_step
			# images
			for rgb_name in self.rgb_list + self.add_img:
				path_ = self.save_path / rgb_name / ('%04d.png' % frame)
				Image.fromarray(tick_data[rgb_name]).save(path_)
				data_row_list.append(str(path_))
			# lidar
			for i, rgb_name in enumerate(self.lidar_list):
				path_ = self.save_path / rgb_name / ('%04d.png' % frame)
				Image.fromarray(cm.gist_earth(tick_data['lidar_processed'][0][0, i], bytes=True)).save(path_)
				data_row_list.append(str(path_))

			# pid_metadata
			pid_metadata = tick_data['pid_metadata']
			path_ = self.save_path / 'pid_metadata' / ('%04d.json' % frame)
			outfile = open(path_, 'w')
			json.dump(pid_metadata, outfile, indent=4)
			outfile.close()
			data_row_list.append(str(path_))

		# collection
		data_row = ','.join([str(i) for i in data_row_list])
		with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
			f_out.write(data_row+'\n')



	# def save(self, tick_data):
	# 	frame = self.step // 10

	# 	Image.fromarray(tick_data['seg_center']).save(self.save_path / 'seg_center' / ('%04d.png' % frame))

	# def destroy(self):
	# 	del self.net
	# 	del self.segmentation_net
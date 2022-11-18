import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from multiprocessing import Process

HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))

def debug_display(tick_data, steer, throttle, brake, step):
    # modification
    _rgb = Image.fromarray(tick_data['rgb'])
    # _draw_rgb = ImageDraw.Draw(_rgb)
    # _draw_rgb.ellipse((target_cam[0]-3,target_cam[1]-3,target_cam[0]+3,target_cam[1]+3), (255, 255, 255))

    # for x, y in out:
    #     x = (x + 1) / 2 * 256
    #     y = (y + 1) / 2 * 144

    # _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))
    hstack_img = []
    if 'rgb_left' in tick_data:
        hstack_img.append(tick_data['rgb_left'])
    hstack_img.append(_rgb)
    if 'rgb_right' in tick_data:
        hstack_img.append(tick_data['rgb_right'])
    _combined = Image.fromarray(np.hstack(hstack_img))
    _combined = _combined.resize((int(256 / _combined.size[1] * _combined.size[0]), 256))

    _topdown = Image.fromarray(tick_data['topdown'])
    _topdown.thumbnail((256, 256))

    _combined = Image.fromarray(np.hstack((_combined, _topdown)))

    _draw = ImageDraw.Draw(_combined)
    _draw.text((5, 10), 'Steer: %.3f' % steer)
    _draw.text((5, 30), 'Throttle: %.3f' % throttle)
    _draw.text((5, 50), 'Brake: %s' % brake)
    _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'])
    _draw.text((5, 90), 'Far Command: %s' % str(tick_data['far_command'].name))
    # _draw.text((5, 110), 'Desired: %.3f' % desired_speed)

    cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)

class Saver(Process):
    def __init__(self, dict_, list_):
        '''
            dict_['save']:
            list_: list of origin frame, like [{source, keys, type, floder},]
        '''
        self.dict_ = dict_
        self.list_ = list_
        self.timestamp_last = self.dict_['timestamp_last']
        super().__init__()

    def run(self):
        from customized_utils2 import end
        print("Saver(Process) run")
        while(self.dict_['save']):
            list_info = [_info for _info in self.list_] # TODO: try
            timestamps = np.array([tick_data['timestamp'] for tick_data in list_info])
            indexs = np.where(timestamps > self.timestamp_last)[0]
            for index in list(indexs):
                tick_data = list_info[index]
                self._save(tick_data)
                self.timestamp_last = tick_data['timestamp']
                self.dict_['timestamp_last'] = self.timestamp_last
            else:
                time.sleep(0.1)
        end()

    def _save(self, x):
        '''call in run only'''
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

from dataclasses import dataclass, field
from typing import Optional

import cv2 as cv
import freenect as fn
import numpy as np
import structlog


LOG = structlog.get_logger()

RESOLUTIONS = (fn.RESOLUTION_LOW, fn.RESOLUTION_MEDIUM, fn.RESOLUTION_HIGH)
VIDEO_FORMATS = (fn.VIDEO_RGB, fn.VIDEO_IR_8BIT)
DEPTH_FORMATS = (fn.DEPTH_10BIT, fn.DEPTH_11BIT, fn.DEPTH_MM, fn.DEPTH_REGISTERED)

# Globals
_ctx = None
_devices = {}


def get_ctx():
    global _ctx
    if not _ctx:
        _ctx = fn.init()
    return _ctx


def get_device(dev_num=0):
    if dev_num in _devices:
        return _devices[dev_num]

    dev = fn.open_device(get_ctx(), dev_num)
    _devices[dev_num] = dev
    return dev


@dataclass
class FreenectMode:
    resolution: int
    fmt: int

    def __post_init__(self):
        self._res_index = RESOLUTIONS.index(self.resolution)

    def cycle_resolution(self):
        self._res_index = (self._res_index + 1) % len(RESOLUTIONS)
        self.resolution = RESOLUTIONS[self._res_index]

    def increase_resolution(self):
        self._res_index = min((self._res_index + 1), (len(RESOLUTIONS) - 1))
        self.resolution = RESOLUTIONS[self._res_index]

    def decrease_resolution(self):
        self._res_index = max((self._res_index - 1), 0)
        self.resolution = RESOLUTIONS[self._res_index]


@dataclass
class DepthMode(FreenectMode):

    def __post_init__(self):
        super().__post_init__()
        self._fmt_index = DEPTH_FORMATS.index(self.fmt)

    def cycle_format(self):
        self._fmt_index = (self._fmt_index + 1) % len(DEPTH_FORMATS)
        self.fmt = DEPTH_FORMATS[self._res_index]


@dataclass
class VideoMode(FreenectMode):

    def __post_init__(self):
        super().__post_init__()
        self._fmt_index = VIDEO_FORMATS.index(self.fmt)

    def cycle_format(self):
        self._fmt_index = (self._fmt_index + 1) % len(VIDEO_FORMATS)
        self.fmt = VIDEO_FORMATS[self._res_index]


class DeviceController:

    def __init__(self, device_num: int = 0,
                 video_mode: Optional[VideoMode] = None,
                 depth_mode: Optional[DepthMode] = None):
        self.device_num = device_num
        self.device = None
        self.video_mode = video_mode or VideoMode(fn.RESOLUTION_HIGH, fn.VIDEO_IR_8BIT)
        self.depth_mode = depth_mode or DepthMode(fn.RESOLUTION_HIGH, fn.DEPTH_11BIT)
        self._should_kill = False
        self._actions = {
            'q': self.kill,
            'v': self.cycle_video,
            'd': self.cycle_depth,
            'r': self.cycle_resolution,
            '+': self.increase_resolution,
            '=': self.increase_resolution,
            '-': self.decrease_resolution
        }
        self.log = LOG.bind(device_num=self.device_num,
                            video_mode=self.video_mode,
                            depth_mode=self.depth_mode)

    def set_video_mode(self):
        self.log.debug('Setting video mode')
        fn.set_video_mode(self.device, self.video_mode.resolution, self.video_mode.fmt)

    def set_depth_mode(self):
        self.log.debug('Setting depth mode')
        fn.set_depth_mode(self.device, self.depth_mode.resolution, self.depth_mode.fmt)

    def set_modes(self):
        self.set_video_mode()
        self.set_depth_mode()

    def cycle_video(self):
        self.video_mode.cycle_format()
        self.set_video_mode()

    def cycle_depth(self):
        self.depth_mode.cycle_format()
        self.set_depth_mode()

    def cycle_resolution(self):
        self.video_mode.cycle_resolution()
        self.depth_mode.cycle_resolution()
        self.set_modes()

    def increase_resolution(self):
        self.video_mode.increase_resolution()
        self.depth_mode.increase_resolution()
        self.set_modes()

    def decrease_resolution(self):
        self.video_mode.decrease_resolution()
        self.depth_mode.decrease_resolution()
        self.set_modes()

    def kill(self):
        self._should_kill = True

    def waitKey(self):
        keycode = cv.waitKey(1)
        if keycode > -1:
            char = chr(keycode).lower()
            LOG.debug('Pressed key', keycode=keycode, char_lower=char)
            if char in self._actions:
                self._actions[char]()

    def display(self, video=True, depth=False):
        if not self.device:
            self.device = get_device(self.device_num)

        if video:
            self.set_video_mode()
        if depth:
            self.set_depth_mode()

        def video_cb(dev, video, timestamp):
            cv.imshow(f'Video - Device {self.device_num}', video)
            self.waitKey()

        def depth_cb(dev, depth, timestamp):
            pass

        def body_cb(dev, ctx):
            if self._should_kill:
                LOG.info('Killing displays')
                raise fn.Kill

        on_video = video_cb if video else None
        on_depth = depth_cb if depth else None

        fn.runloop(depth=on_depth, video=on_video, body=body_cb, dev=self.device)


if __name__ == '__main__':
    controller = DeviceController()
    controller.display()

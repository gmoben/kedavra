from dataclasses import dataclass, field
from typing import Optional

import cv2 as cv
import freenect as fn
import numpy as np
import structlog


LOG = structlog.get_logger()

# Defined here, but let's validate before trying to call `set_XXX_mode` in the first place.
# https://github.com/OpenKinect/libfreenect/blob/master/src/cameras.c#L41-L74
#
# Formats supported by the C++ API but not the Cython wrapper are commented out
# (although this would be easy to implement...)
# https://github.com/OpenKinect/libfreenect/blob/master/wrappers/python/freenect.pyx#L508
# https://github.com/OpenKinect/libfreenect/blob/master/wrappers/python/freenect.pyx#L538-L548
SUPPORTED_VIDEO_MODES = {
    fn.RESOLUTION_HIGH: (
        fn.VIDEO_RGB,
        # fn.VIDEO_BAYER,
        fn.VIDEO_IR_8BIT,
        fn.VIDEO_IR_10BIT,
        # fn.VIDEO_IR_10BIT_PACKED,
    ),
    fn.RESOLUTION_MEDIUM: (
        fn.VIDEO_RGB,
        # fn.VIDEO_BAYER,
        fn.VIDEO_IR_8BIT,
        fn.VIDEO_IR_10BIT,
        # fn.VIDEO_IR_10BIT_PACKED,
        # fn.VIDEO_YUV_RGB,
        # fn.VIDEO_YUV_RAW
    )
}
SUPPORTED_DEPTH_MODES = {
    fn.RESOLUTION_MEDIUM: (
        fn.DEPTH_11BIT,
        fn.DEPTH_10BIT,
        # fn.DEPTH_11BIT_PACKED,
        # fn.DEPTH_10BIT_PACKED,
        fn.DEPTH_REGISTERED,
        fn.DEPTH_MM
    )
}


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
        if not hasattr(self, 'SUPPORTED_MODES'):
            raise AttributeError('SUPPORTED_MODES must be assigned in the subclass')
        self.validate()
        self.SUPPORTED_RESOLUTIONS = sorted(list(self.SUPPORTED_MODES.keys()))

    def validate(self):
        try:
            assert self.resolution in self.SUPPORTED_MODES
        except AssertionError:
            LOG.error('Unsupported resolution', resolution=self.resolution,
                      supported=list(self.SUPPORTED_MODES.keys()))
            raise
        try:
            assert self.fmt in self.SUPPORTED_MODES[self.resolution]
        except AssertionError:
            LOG.error('Unsupported format for resolution', resolution=self.resolution, format=self.fmt,
                      supported_formats=list(self.SUPPORTED_MODES[self.resolution]))
            raise

    def cycle_resolution(self):
        index = self.SUPPORTED_RESOLUTIONS.index(self.resolution)
        index = (index + 1) % len(self.SUPPORTED_RESOLUTIONS)
        self.resolution = self.SUPPORTED_RESOLUTIONS[index]
        self.validate()

    def increase_resolution(self):
        index = self.SUPPORTED_RESOLUTIONS.index(self.resolution)
        index = min((index + 1), (len(self.SUPPORTED_RESOLUTIONS) - 1))
        self.resolution = self.SUPPORTED_RESOLUTIONS[index]
        self.validate()

    def decrease_resolution(self):
        index = self.SUPPORTED_RESOLUTIONS.index(self.resolution)
        index = max((index - 1), 0)
        self.resolution = self.SUPPORTED_RESOLUTIONS[index]
        self.validate()

    def cycle_format(self):
        index = self.SUPPORTED_MODES[self.resolution].index(self.fmt)
        index = (index + 1) % len(self.SUPPORTED_MODES[self.resolution])
        self.fmt = self.SUPPORTED_MODES[self.resolution][index]
        self.validate()


@dataclass
class DepthMode(FreenectMode):

    def __post_init__(self):
        self.SUPPORTED_MODES = SUPPORTED_DEPTH_MODES
        super().__post_init__()


@dataclass
class VideoMode(FreenectMode):

    def __post_init__(self):
        self.SUPPORTED_MODES = SUPPORTED_VIDEO_MODES
        super().__post_init__()


class DeviceController:

    def __init__(self, device_num: int = 0,
                 video_mode: Optional[VideoMode] = None,
                 depth_mode: Optional[DepthMode] = None):
        self.device_num = device_num
        self.device = None
        self.video_mode = video_mode or VideoMode(fn.RESOLUTION_HIGH, fn.VIDEO_IR_8BIT)
        self.depth_mode = depth_mode or DepthMode(fn.RESOLUTION_MEDIUM, fn.DEPTH_11BIT)
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
        # This doesn't work right now unfortunately
        # TODO: Add reset function to stop streams, close and recreate the device,
        # set the modes, and run display again. Might need to avoid using `runloop`
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
                LOG.info('Killing streams')
                raise fn.Kill

        on_video = video_cb if video else None
        on_depth = depth_cb if depth else None

        fn.runloop(depth=on_depth, video=on_video, body=body_cb, dev=self.device)


if __name__ == '__main__':
    controller = DeviceController()
    controller.display()

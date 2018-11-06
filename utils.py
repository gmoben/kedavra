import cv2 as cv
import freenect as fn
import numpy as np
import structlog


LOG = structlog.get_logger()

# Globals
should_kill = False
_ctx = None
_devices = {}


def make_device(device_num=0):
    global _ctx

    if device_num in _devices:
        return _devices[device_num]

    if not _ctx:
        _ctx = fn.init()

    dev = fn.open_device(_ctx, device_num)
    _devices[device_num] = dev
    return dev


def _waitKey():
    global should_kill
    keycode = cv.waitKey(1)
    if keycode > -1:
        char = chr(keycode).lower()
        LOG.debug('Pressed key', keycode=keycode, char_lower=char)
        if char == 'q':
            should_kill = True


def _video_cb(dev, video, timestamp):
    cv.imshow('Video', video)
    _waitKey()


def _depth_cb(dev, depth, timestamp):
    pass


def _body_cb(dev, ctx):
    global should_kill
    if should_kill:
        LOG.info('Killing application')
        raise fn.Kill


def display(video_res=fn.RESOLUTION_HIGH,
            video_fmt=fn.VIDEO_IR_8BIT,
            depth_res=fn.RESOLUTION_HIGH,
            depth_fmt=fn.DEPTH_11BIT,
            video=True, depth=False):
    dev = make_device()

    if video:
        fn.set_video_mode(dev, video_res, video_fmt)
    if depth:
        fn.set_depth_mode(dev, depth_res, depth_fmt)

    on_video = _video_cb if video else None
    on_depth = _depth_cb if depth else None

    fn.runloop(depth=on_depth, video=on_video, body=_body_cb, dev=dev)


display()

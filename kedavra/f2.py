import os
import threading
import time
from queue import Queue

import cv2 as cv
import freenect2 as fn2
import numpy as np
import structlog
from skimage.transform import resize


LOG = structlog.get_logger()

ir_queue = Queue()

THRESHOLD = 150
KEYPOINT_TIMEOUT = 1
DELAY_AFTER_RESET = 1.5
VELOCITY_LOWER_BOUND = 10
VELOCITY_UPPER_BOUND = 120
MIN_POINTS = 20


def create_blob_detector():
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = THRESHOLD
    params.maxThreshold = 255

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = 0.01
    params.maxArea = 100

    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByConvexity = True
    params.minConvexity = 0.1

    params.filterByInertia = False

    return cv.SimpleBlobDetector_create(params)


detector = create_blob_detector()
mog2 = cv.createBackgroundSubtractorMOG2(
    history=10, varThreshold=500)


points = []
trace = None
last_point_ts = None
last_reset_ts = None
ready = False


def append_point(point):
    global last_point_ts
    if not points:
        points.append((point[0], point[1]))
    else:
        prev_point = points[-1]
        norm = cv.norm(prev_point - point)
        if not VELOCITY_LOWER_BOUND <= norm <= VELOCITY_UPPER_BOUND:
            return
        # LOG.debug('Velocity', norm=norm)
        points.append((point[0], point[1]))
    last_point_ts = time.process_time()


def reset_trace(shape, save=True, outdir='/code/ben/kedavra/training'):
    global points
    global trace
    global last_point_ts
    global last_reset_ts
    path = None
    if save:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        time_ns = time.time_ns()
        path = f'{outdir}/{time_ns}.bmp'
        num_points = len(points)
        if num_points >= MIN_POINTS:
            LOG.debug('Writing trace', path=path, num_points=num_points)
            cv.imwrite(path, trace)
        else:
            LOG.debug('Not enough points to write trace', num_points=num_points)
    LOG.debug('Resetting trace', save=save)
    points = []
    trace = np.zeros(shape, dtype=np.uint8)
    last_point_ts = None
    last_reset_ts = time.process_time()


def worker():
    global ready
    while True:
        image = ir_queue.get(timeout=3)
        image *= 255
        image = image.astype(np.uint8)

        thresholded = image.copy()
        thresholded[thresholded < THRESHOLD] = 0

        mask = mog2.apply(thresholded)

        now = time.process_time()
        if last_point_ts is not None:
            elapsed_since_point = now - last_point_ts
        else:
            elapsed_since_point = 0

        if trace is None:
            reset_trace(mask.shape, save=False)
        elif elapsed_since_point >= KEYPOINT_TIMEOUT:
            reset_trace(mask.shape)

        elapsed_since_reset = now - last_reset_ts

        keypoints = detector.detect(mask)

        if elapsed_since_reset >= DELAY_AFTER_RESET:
            if ready is False:
                ready = True
                LOG.info('Ready to record')
            for p in cv.KeyPoint_convert(keypoints):
                append_point(p)

            if len(points) >= 2:
                p1 = points[-2]
                p2 = points[-1]
                cv.line(trace, p1, p2, (255, 255, 255), 2)
        else:
            ready = False

        black = np.zeros(mask.shape, dtype=np.uint8)

        image_with_keypoints = cv.drawKeypoints(black, keypoints, np.array([]),
                                                (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        overlayed = cv.drawKeypoints(image, keypoints, np.array([]),
                                     (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv.imshow('Keypoints', image_with_keypoints)
        cv.imshow('Overlayed', overlayed)
        cv.imshow('Trace', trace)
        ir_queue.task_done()


def scale(image, factor):
    # This is too slow to actually use without hardware acceleration
    return resize(image, (image.shape[0] * factor, image.shape[1] * factor))


class DeviceController:

    def __init__(self, show_color=False, show_ir=True, frame_timeout=3):
        self._device = fn2.Device()
        self._show_color = show_color
        self._show_ir = show_ir
        self._frame_timeout = frame_timeout
        self._should_kill = False
        self._threshold = 0.5
        self._actions = {
            'q': self.kill
        }
        self.ir_thread = threading.Thread(target=worker)
        self._bg_init_framecount = None

    def kill(self):
        self._should_kill = True

    def waitKey(self):
        keycode = cv.waitKey(1)
        if keycode > -1:
            char = chr(keycode).lower()
            LOG.debug('Pressed key', keycode=keycode, char_lower=char)
            if char in self._actions:
                self._actions[char]()

    def preprocess_ir(self, image):
        image /= image.max()
        return image

    def on_frame(self, frame_type, frame):
        image = frame.to_array().copy()
        if self._show_color and frame_type == fn2.FrameType.Color:
            cv.imshow('Color', image)
        elif self._show_ir and frame_type == fn2.FrameType.Ir:
            image = self.preprocess_ir(image)
            ir_queue.put(image)
            cv.imshow('IR', image)

    def restart(self):
        self._device.close()
        self._device = fn2.Device()
        self._device.start()

    def cleanup(self):
        self._device.stop()
        self._device.close()
        self.ir_thread.join()

    def display(self):
        self.ir_thread.start()
        self._device.start()
        retries = 3
        while not self._should_kill:
            try:
                frame_type, frame = self._device.get_next_frame(self._frame_timeout)
                self.on_frame(frame_type, frame)
                self.waitKey()
            except fn2.NoFrameReceivedError:
                LOG.error('Timed out waiting for new frame',
                          timeout=self._frame_timeout)
                if retries:
                    LOG.info('Restarting device', retries=retries)
                    self.restart()
                    retries -= 1
                else:
                    LOG.info('Killing device')
                    self.kill()
            except Exception:
                LOG.exception()
                self.kill()
        self.cleanup()
        cv.destroyAllWindows()


if __name__ == '__main__':
    controller = DeviceController()
    controller.display()

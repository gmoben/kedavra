import threading
from queue import Queue

import cv2 as cv
import freenect2 as fn2
import numpy as np
import structlog
from skimage.transform import resize


LOG = structlog.get_logger()

ir_queue = Queue()


def create_blob_detector():
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 150
    params.maxThreshold = 255

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = 0.5
    params.maxArea = 100

    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByConvexity = True
    params.minConvexity = 0.5

    params.filterByInertia = False

    return cv.SimpleBlobDetector_create(params)


detector = create_blob_detector()
mog2 = cv.createBackgroundSubtractorMOG2(
    history=10, varThreshold=10)


def worker():
    while True:
        image = ir_queue.get(timeout=3)
        image *= 255
        image = image.astype(np.uint8)
        image[image < 150] = 0
        mask = mog2.apply(image)

        keypoints = detector.detect(mask)
        black = np.zeros(mask.shape, dtype=np.uint8)
        image_with_keypoints = cv.drawKeypoints(black, keypoints, np.array([]),
                                                (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('Keypoints', image_with_keypoints)
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

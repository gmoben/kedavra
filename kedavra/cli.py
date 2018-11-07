from typing import Any, List, Optional

import fire
import structlog

from kedavra.utils import (
    DEFAULT_VIDEO_MODES,
    DeviceController,
    num_devices
)


LOG = structlog.get_logger()


class InvalidSelection(Exception):
    pass


def select_option(options: List[Any], prompt_text: str = 'Select an option',
                  default: Optional[int] = None):
    """Display a selectable list of options"""
    prompt_text = prompt_text.rstrip(':') + ':'

    def get_selection():
        print(prompt_text)
        for idx, option in enumerate(options):
            print(f"({idx}): {option}".format(idx, option))
        prompt = "Enter selection"
        if default is not None:
            prompt += f" [{default}]"
        prompt += ': '
        i = input(prompt)
        print(f"Selected {i}")
        if i == '':
            if default is not None:
                return options[default]
            else:
                print('Must select a value. Try again...')
                raise InvalidSelection

        try:
            index = int(i)
            if not index < len(options):
                raise ValueError
        except ValueError:
            print('Invalid selection. Try again...')
            raise InvalidSelection
        else:
            return options[index]

    while True:
        try:
            return get_selection()
        except InvalidSelection:
            continue


class KinectCLI:
    """Fire CLI to interact with a Kinect (v1) via libfreenect"""

    def _get_device_num(self, device_num: Optional[int] = None):
        """Validate or prompt for a device number"""
        n_devices = num_devices()
        if n_devices == 0:
            LOG.error('No devices found')
            exit(1)

        if device_num is None:
            device_num = select_option(range(n_devices),
                                       "Select a device", default=0)

        if device_num >= n_devices:
            LOG.error('Device selection out of range',
                      device_num=device_num, num_devices=num_devices)
            exit(1)

        return device_num

    def display(self, device_num: Optional[int] = None,
                video_source: Optional[str] = None, video: bool = True,
                depth: bool = False):
        """Launch GUI(s) for video and/or depth inputs"""
        device_num = self._get_device_num(device_num)

        if video_source is None:
            options = sorted(list(DEFAULT_VIDEO_MODES.keys()))
            video_source = select_option(options, "Select a video mode",
                                         default=options.index('ir'))

        try:
            video_source = video_source.lower()
            video_mode = DEFAULT_VIDEO_MODES[video_source]
        except KeyError:
            LOG.error('Invalid video source', video_source=video_source,
                      default_video_modes=DEFAULT_VIDEO_MODES)
            exit(1)

        LOG.debug('Video mode', source=video_source, mode=video_mode)
        controller = DeviceController(device_num, video_mode=video_mode)
        controller.display(video, depth)


def main():
    fire.Fire(KinectCLI)


if __name__ == '__main__':
    main()

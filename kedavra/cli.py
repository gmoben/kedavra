from typing import Any, List, Optional

import fire

from kedavra.utils import (
    IR_MODE,
    RGB_MODE,
    DeviceController,
    num_devices
)


class InvalidSelection(Exception):
    pass


def select_option(options: List[Any], prompt_text: str = 'Select an option',
                  default: Optional[int] = None):

    prompt_text = prompt_text.rstrip(':') + ':'

    def get_selection():
        print(prompt_text)
        for idx, option in enumerate(options):
            print(f"({idx}): {option}".format(idx, option))
        prompt = "Enter selection"
        if default is not None:
            prompt += f" ({default})"
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


class Cli:

    def get_device_num(self, device_num: Optional[int] = None):
        n_devices = num_devices()
        if n_devices == 0:
            print('No devices found')
            exit(1)

        if device_num is None:
            device_num = select_option(range(n_devices),
                                       "Select a device", default=0)

        if device_num >= n_devices:
            print('Device selection out of range')
            exit(1)

        return device_num

    def display(self, device_num: Optional[int] = None,
                video_type: str = 'ir', video: bool = True,
                depth: bool = False):

        device_num = self.get_device_num(device_num)

        video_modes = {
            'ir': IR_MODE,
            'rgb': RGB_MODE
        }

        try:
            video_mode = video_modes[video_type]
        except KeyError:
            print(f"Invalid video type '{video_type}'")
            exit(1)

        controller = DeviceController(device_num, video_mode=video_mode)
        controller.display(video, depth)


if __name__ == '__main__':
    fire.Fire(Cli)

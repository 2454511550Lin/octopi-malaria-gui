from microscope import Microscope
import time

microscope = Microscope(is_simulation=False)

microscope.camera.start_streaming()
microscope.camera.set_software_triggered_acquisition()
microscope.camera.disable_callback()

microscope.home_xyz()
microscope.to_scanning_position()

for i in range(5):
    for channel in ["BF LED matrix left half","BF LED matrix right half","Fluorescence 405 nm Ex"]:
        microscope.set_channel(channel)
        image = microscope.acquire_image()
        print(image)

# microscope.start_live()
# time.sleep(2)
# microscope.stop_live()

microscope.close()

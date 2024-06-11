import time
import threading
from typing import Optional

import torch.nn as nn
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.viewer_elements import ViewerControl
from nerfstudio.cameras.cameras import Cameras

class CameraLogger:
    def __init__(self, viewer_control: ViewerControl, log_interval: int = 5):
        """
        Initializes the CameraLogger.
        
        Args:
            viewer_control (ViewerControl): The viewer control object.
            log_interval (int): Interval in seconds between logging camera positions.
        """
        self.viewer_control = viewer_control
        self.log_interval = log_interval
        self.running = False

    def get_camera_position(self, img_height: int, img_width: int, client_id: Optional[int] = None):
        """
        Retrieves the camera position from the viewer.

        Args:
            img_height (int): Height of the image.
            img_width (int): Width of the image.
            client_id (Optional[int]): Client ID (if applicable).

        Returns:
            Optional[torch.Tensor]: Camera extrinsics matrix (3x4) if the viewer is connected, otherwise None.
        """
        camera: Optional[Cameras] = self.viewer_control.get_camera(img_height, img_width, client_id)
        if camera is None:
            return None
        return camera.camera_to_worlds[0, ...]

    def log_camera_position(self):
        """
        Logs the camera position at regular intervals.
        """
        while self.running:
            position = self.get_camera_position(100, 100)
            if position is not None:
                CONSOLE.log(f"Camera Position: {position}")
                # Here you could save to a file instead of printing
                with open('camera_positions.txt', 'a') as f:
                    f.write(f"{position.cpu().numpy()}\n")
            else:
                CONSOLE.log("Viewer not connected.")
            time.sleep(self.log_interval)

    def start_logging(self):
        """
        Starts logging the camera position.
        """
        CONSOLE.log("Logging camera position.")
        self.running = True
        self.thread = threading.Thread(target=self.log_camera_position)
        self.thread.start()

    def stop_logging(self):
        """
        Stops logging the camera position.
        """
        self.running = False
        self.thread.join()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.viewer_control = ViewerControl()
        self.camera_logger = CameraLogger(self.viewer_control)

    def start_logging(self):
        self.camera_logger.start_logging()

    def stop_logging(self):
        self.camera_logger.stop_logging()

if __name__ == "__main__":
    model = MyModel()
    try:
        model.start_logging()
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        model.stop_logging()
        CONSOLE.log("Logging stopped.")

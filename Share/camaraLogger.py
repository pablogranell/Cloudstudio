from os import wait
from Cloudstudio.nerfstudio.viewer.viewer_elements import ViewerControl


viewer_control = ViewerControl()._setup()
while True :
    wait(1)
    cameraPosition = viewer_control.get_camera(100, 100)
    if cameraPosition is not None:
        print(f"Camera Position: {cameraPosition}")
    else:
        print("Viewer not connected.")

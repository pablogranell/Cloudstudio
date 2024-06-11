from os import wait
from viewer_elements import ViewerElements


viewer_control = ViewerElements.ViewerControl._setup()
while True :
    wait(1)
    cameraPosition = viewer_control.get_camera(100, 100)
    if cameraPosition is not None:
        print(f"Camera Position: {cameraPosition}")
    else:
        print("Viewer not connected.")

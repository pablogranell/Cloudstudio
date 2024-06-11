from os import wait
#from Cloudstudio import nerfstudio
#from nerfstudio import viewer
#from nerfstudio.viewer import viewer_elements
from nerfstudio.viewer.viewer_elements import ViewerControl


viewer_control = ViewerControl()
while True :
    wait(1)
    cameraPosition = viewer_control.get_camera(100, 100)
    if cameraPosition is not None:
        print(f"Camera Position: {cameraPosition}")
    else:
        print("Viewer not connected.")

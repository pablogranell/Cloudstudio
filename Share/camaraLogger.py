from os import wait
import time
#from Cloudstudio import nerfstudio
#from nerfstudio import viewer
#from nerfstudio.viewer import viewer_elements
from nerfstudio.viewer.viewer_elements import ViewerControl
from viser import ViserServer

viser_server = ViserServer()
viewer_control = ViewerControl()
while True :
    time.sleep(0.01)
    if ViserServer.get_clients() == 0:
        time.sleep(0.01)
        print("No clients connected.")
    else:
        cameraPosition = viewer_control.get_camera(100, 100)
        if cameraPosition is not None:
            print(f"Camera Position: {cameraPosition}")
        else:
            print("Viewer not connected.")

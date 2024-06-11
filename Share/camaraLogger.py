import time

from nerfstudio.viewer.viewer_elements import ViewerControl
import nerfstudio.viewer.viewer as viewer

mainViewer = viewer.Viewer()
viewer_server = viewer.ViewerServer(mainViewer)
viewer_control = ViewerControl(viewer_server)
while True :
    time.sleep(0.01)
    #print("Waiting for clients to connect.")
    if viewer_control.get_num_clients() == 0:
        time.sleep(0.01)
        print("No clients connected.")
    else:
        cameraPosition = viewer_control.get_camera(100, 100)
        if cameraPosition is not None:
            print(f"Camera Position: {cameraPosition}")
        else:
            print("Viewer not connected.")

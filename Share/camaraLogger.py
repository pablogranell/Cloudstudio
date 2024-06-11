import time

from nerfstudio.viewer.viewer_elements import ViewerControl
import nerfstudio.viewer.viewer as viewer

viewer_server = viewer.ViewerServer()
clients = viewer_server.get_clients()
for client_id, client_handle in clients.items():
    print(f"Client {client_id} is connected with handle {client_handle}")
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

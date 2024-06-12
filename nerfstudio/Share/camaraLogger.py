from torck import nn
from nerfstudio.viewer.viewer_elements import ViewerControl

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inicia el controlador del visor
        test = ViewerControl()
        while not test.is_connected():
            time.sleep(1)
            print("Esperando conexión con el visor...")
        print("prueba")
        print(test.get_camera(100, 100))
       
        camera = self.viewer_control.get_camera(100, 100)
        if camera is None:
            print("Viewer no está conectado aún.")
            return

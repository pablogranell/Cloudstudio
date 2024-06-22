"""Manage the state of the viewer"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE
import viser
import viser.theme
import viser.transforms as vtf
from typing_extensions import assert_never

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nerfstudio.viewer.control_panel import ControlPanel
from nerfstudio.viewer.export_panel import populate_export_tab
from nerfstudio.viewer.render_panel import populate_render_tab
from nerfstudio.viewer.render_state_machine import RenderAction, RenderStateMachine
from nerfstudio.viewer.utils import CameraState, parse_object
from nerfstudio.viewer.viewer_elements import ViewerControl, ViewerElement
from nerfstudio.viewer_legacy.server import viewer_utils

if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer


VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0
sincronizacion = False
control = 0


def toggle_sincronizacion():
    global sincronizacion
    sincronizacion = not sincronizacion
    CONSOLE.print(f"Sincronizacion: {sincronizacion}")

def toggle_control(num_clientes):
    global control
    control += 1
    if control >= num_clientes:
        control = 0
    CONSOLE.print(f"Control: {control}")

@decorate_all([check_main_thread])
class Viewer:
    """Class to hold state for viewer variables

    Args:
        config: viewer setup configuration
        log_filename: filename to log viewer output to
        datapath: path to data
        pipeline: pipeline object to use
        trainer: trainer object to use
        share: print a shareable URL

    Attributes:
        viewer_info: information string for the viewer
        viser_server: the viser server
    """

    viewer_info: List[str]
    viser_server: viser.ViserServer

    def __init__(
        self,
        config: cfg.ViewerConfig,
        log_filename: Path,
        datapath: Path,
        pipeline: Pipeline,
        trainer: Optional[Trainer] = None,
        train_lock: Optional[threading.Lock] = None,
        share: bool = False,
        clientN = 0,
    ):
        self.ready = False  # Set to True at end of constructor.
        self.config = config
        self.trainer = trainer
        self.last_step = 0
        self.train_lock = train_lock
        self.pipeline = pipeline
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath
        self.include_time = self.pipeline.datamanager.includes_time
        self.clientN = clientN

        if self.config.websocket_port is None:
            websocket_port = viewer_utils.get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        # viewer specific variables
        self.output_type_changed = True
        self.output_split_type_changed = True
        self.step = 0
        self.train_btn_state: Literal["training", "paused", "completed"] = "training"
        self._prev_train_state: Literal["training", "paused", "completed"] = "training"
        self.last_move_time = 0

        self.viser_server = viser.ViserServer(host=config.websocket_host, port=websocket_port)
        # Set the name of the URL either to the share link if available, or the localhost
        share_url = None
        if share:
            share_url = self.viser_server.request_share_url()
            if share_url is None:
                print("Couldn't make share URL!")

        if share_url is not None:
            self.viewer_info = [f"Viewer at: http://localhost:{websocket_port} or {share_url}"]
        elif config.websocket_host == "0.0.0.0":
            # 0.0.0.0 is not a real IP address and was confusing people, so
            # we'll just print localhost instead. There are some security
            # (and IPv6 compatibility) implications here though, so we should
            # note that the server is bound to 0.0.0.0!
            self.viewer_info = [f"Viewer running locally at: http://localhost:{websocket_port} (listening on 0.0.0.0)"]
        else:
            self.viewer_info = [f"Viewer running locally at: http://{config.websocket_host}:{websocket_port}"]

        self.viser_server.configure_theme(
            control_layout="floating",
            dark_mode=True,
            brand_color=(0, 51, 204),
            show_logo=False,
            show_share_button=False,
        )

        self.render_statemachines: Dict[int, RenderStateMachine] = {}
        self.viser_server.on_client_disconnect(self.handle_disconnect)
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_connect(self.sync_camera)
        # Populate the header, which includes the pause button, train cam button, and stats
        self.pause_train = self.viser_server.add_gui_button(
            label="Pausar Entrenamiento", disabled=False, icon=viser.Icon.PLAYER_PAUSE_FILLED
        )
        self.pause_train.on_click(lambda _: self.toggle_pause_button())
        self.pause_train.on_click(lambda han: self._toggle_training_state(han))
        self.resume_train = self.viser_server.add_gui_button(
            label="Continuar Entrenamiento", disabled=False, icon=viser.Icon.PLAYER_PLAY_FILLED,
        )
        self.resume_train.on_click(lambda _: self.toggle_pause_button())
        self.resume_train.on_click(lambda han: self._toggle_training_state(han))
        self.resume_train.visible = False

        #Add button to sync camera position to other clients
        self.sync_camera = self.viser_server.add_gui_button(
            label="Activar sincronizacion", disabled=False, icon=viser.Icon.CAMERA_SHARE
        )
        self.sync_camera.on_click(lambda _: self.toggle_sync_button())
        self.sync_camera.on_click(lambda _: toggle_sincronizacion())
        self.disable_sync_camera = self.viser_server.add_gui_button(
            label="Desactivar Sincronizacion", disabled=False, icon=viser.Icon.CAMERA_SHARE
        )
        self.disable_sync_camera.on_click(lambda _: self.toggle_sync_button())
        self.disable_sync_camera.on_click(lambda _: toggle_sincronizacion())
        self.disable_sync_camera.visible = False

        # Add button to change control to other clients
        self.change_control = self.viser_server.add_gui_button(
            label="Cambiar Control", disabled=False, icon=viser.Icon.CAMERA_ROTATE
        )
        self.change_control.on_click(lambda _: toggle_control(len(self.viser_server.get_clients())))
        mkdown = self.make_stats_markdown(0, "0x0px", 0, 0)
        self.stats_markdown = self.viser_server.add_gui_markdown(mkdown)
        tabs = self.viser_server.add_gui_tab_group()
        visor = tabs.add_tab("Visor", viser.Icon.FOCUS)
        control_tab = tabs.add_tab("Ajustes", viser.Icon.SETTINGS)
        with control_tab:
            self.control_panel = ControlPanel(
                self.viser_server,
                self.include_time,
                VISER_NERFSTUDIO_SCALE_RATIO,
                self._trigger_rerender,
                self._output_type_change,
                self._output_split_type_change,
                default_composite_depth=False,
            )
        config_path = self.log_filename.parents[0] / "config.yml"
        viewer_gui_folders = dict()

        def prev_cb_wrapper(prev_cb):
            # We wrap the callbacks in the train_lock so that the callbacks are thread-safe with the
            # concurrently executing render thread. This may block rendering, however this can be necessary
            # if the callback uses get_outputs internally.
            def cb_lock(element):
                with self.train_lock if self.train_lock is not None else contextlib.nullcontext():
                    prev_cb(element)

            return cb_lock

        def nested_folder_install(folder_labels: List[str], prev_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [prev_cb_wrapper(prev_cb)(element), self._trigger_rerender()]
            else:
                folder_path = "/".join(prev_labels + [folder_labels[0]])
                if folder_path not in viewer_gui_folders:
                    viewer_gui_folders[folder_path] = self.viser_server.add_gui_folder(folder_labels[0])
                with viewer_gui_folders[folder_path]:
                    nested_folder_install(folder_labels[1:], prev_labels + [folder_labels[0]], element)

        with control_tab:
            from nerfstudio.viewer_legacy.server.viewer_elements import ViewerElement as LegacyViewerElement

            if len(parse_object(pipeline, LegacyViewerElement, "Custom Elements")) > 0:
                from nerfstudio.utils.rich_utils import CONSOLE

                CONSOLE.print(
                    "Legacy ViewerElements detected in model, please import nerfstudio.viewer.viewer_elements instead",
                    style="bold yellow",
                )
            self.viewer_elements = []
            self.viewer_elements.extend(parse_object(pipeline, ViewerElement, "Custom Elements"))
            for param_path, element in self.viewer_elements:
                folder_labels = param_path.split("/")[:-1]
                nested_folder_install(folder_labels, [], element)

            # scrape the trainer/pipeline for any ViewerControl objects to initialize them
            self.viewer_controls: List[ViewerControl] = [
                e for (_, e) in parse_object(pipeline, ViewerControl, "Custom Elements")
            ]
        for c in self.viewer_controls:
            c._setup(self)

        # Diagnostics for Gaussian Splatting: where the points are at the start of training.
        # This is hidden by default, it can be shown from the Viser UI's scene tree table.
        if isinstance(pipeline.model, SplatfactoModel):
            self.viser_server.add_point_cloud(
                "/gaussian_splatting_initial_points",
                points=pipeline.model.means.numpy(force=True) * VISER_NERFSTUDIO_SCALE_RATIO,
                colors=(255, 0, 0),
                point_size=0.01,
                point_shape="circle",
                visible=False,  # Hidden by default.
            )
        self.ready = True

    def toggle_pause_button(self) -> None:
        self.pause_train.visible = not self.pause_train.visible
        self.resume_train.visible = not self.resume_train.visible

    def toggle_cameravis_button(self) -> None:
        self.hide_images.visible = not self.hide_images.visible
        self.show_images.visible = not self.show_images.visible

    def toggle_sync_button(self) -> None:
        self.sync_camera.visible = not self.sync_camera.visible
        self.disable_sync_camera.visible = not self.disable_sync_camera.visible

    def sync_camera(self, client: viser.ClientHandle) -> None:
        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            if sincronizacion:
                clients = self.viser_server.get_clients()
                if client.client_id == control and len(clients) > 1:
                    for id in clients:
                        if id != control:
                            self.last_move_time = time.time()
                            with self.viser_server.atomic():
                                camera_state = self.get_camera_state(clients[control])
                                self.render_statemachines[id].action(RenderAction("move", camera_state))
                                clients[id].camera.position = clients[control].camera.position
                                clients[id].camera.wxyz = clients[control].camera.wxyz

    def controlCustom(self, client: viser.ClientHandle) -> None:
        if client.client_id == 0:
            #Only show stats for the main client
            self.stats_markdown.visible = True
            #Only show the pause training button for the main client
            self.pause_train.visible = True
            #Only show the control tab for the main client
            self.control_panel.visible = True
            #Only show the reset button for the main client
            self.reset_button.visible = True
        else:
            self.stats_markdown.visible = False
            self.pause_train.visible = False
            self.control_panel.visible = False
            self.reset_button.visible = False

    def make_stats_markdown(self, step: Optional[int], res: Optional[str], controladora: Optional[int], cliente: Optional[int]) -> str:
        # if either are None, read it from the current stats_markdown content
        if step is None:
            step = int(self.stats_markdown.content.split("\n")[0].split(": ")[1])
        if res is None:
            res = (self.stats_markdown.content.split("\n")[1].split(": ")[1]).strip()
        if controladora is None:
            controladora = int(self.stats_markdown.content.split("\n")[2].split(": ")[1])
            #CONSOLE.print(f"Controladora: {controladora}")
        if cliente is None:
            cliente = int(self.stats_markdown.content.split("\n")[3].split(": ")[1])
        controladora = control
        cliente = self.clientN
        return f"Pasos: {step}  \nResolucion: {res}  \nControladora: {controladora+1}  \nClientes: {cliente+1}"
    
    def update_step(self, step):
        """
        Args:
            step: the train step to set the model to
        """
        controladora = control
        self.stats_markdown.content = self.make_stats_markdown(step, None, controladora, self.clientN)

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        R = vtf.SO3(wxyz=client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(client.camera.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        camera_state = CameraState(
            fov=client.camera.fov,
            aspect=client.camera.aspect,
            c2w=c2w,
            camera_type=CameraType.PERSPECTIVE,
        )
        return camera_state

    def handle_disconnect(self, client: viser.ClientHandle) -> None:
        self.render_statemachines[client.client_id].running = False
        toggle_control(len(self.viser_server.get_clients()))

    def handle_new_client(self, client: viser.ClientHandle) -> None:
        self.render_statemachines[client.client_id] = RenderStateMachine(self, VISER_NERFSTUDIO_SCALE_RATIO, client)
        self.render_statemachines[client.client_id].start()
        self.clientN = client.client_id
        #Cuidado
        #CONSOLE.print(f"Clientes: {clients}")
        #self.controlCustom(self.viser_server.get_clients()[self.clientN])
        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            if not self.ready:
                return
            self.last_move_time = time.time()
            with self.viser_server.atomic():
                camera_state = self.get_camera_state(client)
                self.render_statemachines[client.client_id].action(RenderAction("move", camera_state))

    def set_camera_visibility(self, visible: bool) -> None:
        """Toggle the visibility of the training cameras."""
        with self.viser_server.atomic():
            for idx in self.camera_handles:
                self.camera_handles[idx].visible = visible

    def update_camera_poses(self):
        # TODO this fn accounts for like ~5% of total train time
        # Update the train camera locations based on optimization
        assert self.camera_handles is not None
        if hasattr(self.pipeline.datamanager, "train_camera_optimizer"):
            camera_optimizer = self.pipeline.datamanager.train_camera_optimizer
        elif hasattr(self.pipeline.model, "camera_optimizer"):
            camera_optimizer = self.pipeline.model.camera_optimizer
        else:
            return
        idxs = list(self.camera_handles.keys())
        with torch.no_grad():
            assert isinstance(camera_optimizer, CameraOptimizer)
            c2ws_delta = camera_optimizer(torch.tensor(idxs, device=camera_optimizer.device)).cpu().numpy()
        for i, key in enumerate(idxs):
            # both are numpy arrays
            c2w_orig = self.original_c2w[key]
            c2w_delta = c2ws_delta[i, ...]
            c2w = c2w_orig @ np.concatenate((c2w_delta, np.array([[0, 0, 0, 1]])), axis=0)
            R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
            R = R @ vtf.SO3.from_x_radians(np.pi)
            self.camera_handles[key].position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
            self.camera_handles[key].wxyz = R.wxyz

    def _trigger_rerender(self) -> None:
        """Interrupt current render."""
        if not self.ready:
            return
        clients = self.viser_server.get_clients()
        for id in clients:
            camera_state = self.get_camera_state(clients[id])
            self.render_statemachines[id].action(RenderAction("move", camera_state))

    def _toggle_training_state(self, _) -> None:
        """Toggle the trainer's training state."""
        if self.trainer is not None:
            if self.trainer.training_state == "training":
                self.trainer.training_state = "paused"
            elif self.trainer.training_state == "paused":
                self.trainer.training_state = "training"

    def _output_type_change(self, _):
        self.output_type_changed = False

    def _output_split_type_change(self, _):
        self.output_split_type_changed = False

    def _pick_drawn_image_idxs(self, total_num: int) -> list[int]:
        """Determine indicies of images to display in viewer.

        Args:
            total_num: total number of training images.

        Returns:
            List of indices from [0, total_num-1].
        """
        if self.config.max_num_display_images < 0:
            num_display_images = total_num
        else:
            num_display_images = min(self.config.max_num_display_images, total_num)
        # draw indices, roughly evenly spaced
        return np.linspace(0, total_num - 1, num_display_images, dtype=np.int32).tolist()

    def init_scene(
        self,
        train_dataset: None,
        train_state: Literal["training", "paused", "completed"],
        eval_dataset: Optional[InputDataset] = None,
    ) -> None:
        """Draw some images and the scene aabb in the viewer.
        Args:
            dataset: dataset to render in the scene
            train_state: Current status of training
        """
        # draw the training cameras and images
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}
        image_indices = self._pick_drawn_image_idxs(len(train_dataset))
        for idx in image_indices:
            image = train_dataset[idx]["image"]
            camera = train_dataset.cameras[idx]
            image_uint8 = (image * 255).detach().type(torch.uint8)
            image_uint8 = image_uint8.permute(2, 0, 1)

            # torchvision can be slow to import, so we do it lazily.
            import torchvision

            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)  # type: ignore
            image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            c2w = camera.camera_to_worlds.cpu().numpy()
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handle = self.viser_server.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=float(2 * np.arctan(camera.cx / camera.fx[0])),
                scale=self.config.camera_frustum_scale,
                aspect=float(camera.cx[0] / camera.cy[0]),
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
                visible=False,
            )

            #@camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles[idx] = camera_handle
            self.original_c2w[idx] = c2w

        self.train_state = train_state
        self.train_util = 0.9

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            num_rays_per_batch: number of rays per batch, used during training
        """
        self.step = step

        if len(self.render_statemachines) == 0:
            return
        # this stops training while moving to make the response smoother
        while time.time() - self.last_move_time < 0.1:
            time.sleep(0.05)
        if self.trainer is not None and self.trainer.training_state == "training" and self.train_util != 1:
            if (
                EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
                and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
            ):
                train_s = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                vis_s = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
                train_util = self.train_util
                vis_n = self.control_panel.max_res**2
                train_n = num_rays_per_batch
                train_time = train_n / train_s
                vis_time = vis_n / vis_s

                render_freq = train_util * vis_time / (train_time - train_util * train_time)
            else:
                render_freq = 30
            if step > self.last_step + render_freq:
                self.last_step = step
                clients = self.viser_server.get_clients()
                for id in clients:
                    camera_state = self.get_camera_state(clients[id])
                    if camera_state is not None:
                        self.render_statemachines[id].action(RenderAction("step", camera_state))
                self.update_camera_poses()
                self.update_step(step)

    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_type_changed:
            self.control_panel.update_colormap_options(dimensions, dtype)
            self.output_type_changed = False

    def update_split_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_split_type_changed:
            self.control_panel.update_split_colormap_options(dimensions, dtype)
            self.output_split_type_changed = False

    def get_model(self) -> Model:
        """Returns the model."""
        return self.pipeline.model

    def training_complete(self) -> None:
        """Called when training is complete."""
        self.training_state = "completed"
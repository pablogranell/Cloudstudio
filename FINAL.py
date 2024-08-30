import os
import queue
import shutil
import subprocess
import threading
import time
import webbrowser
from tkinter import filedialog
import nbformat
import customtkinter as ctk
from kaggle.api.kaggle_api_extended import KaggleApi
import requests
import re
import tkinter as tk
from tkinter import ttk
from socketio import Client

# Constants
KAGGLE_JSON_PATH = os.path.expanduser(os.path.join('~', '.kaggle', 'kaggle.json'))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TERMINAL_QUEUE = queue.Queue()
UPDATE_INTERVAL = 1
UPDATE_INTERVAL_LONG = 10
TIMEOUT = 15
PREDEFINED_COMMANDS = {
    "Transferir video al servidor": {
        "command": f"hypershell-copy {os.path.join(SCRIPT_DIR, 'escena.mp4')} token:/kaggle/temp/Cloudstudio/data/nerfstudio/escena.mp4",
        "local": True
    },
    "Preparar escena": {
        "command": "ns-process-data video --data /kaggle/temp/Cloudstudio/data/nerfstudio/escena --output-dir /kaggle/temp/Cloudstudio/data/nerfstudio/poster",
        "local": False
    },
    "Entrenamiento": {
        "command": "ns-train nerfacto --logging.local-writer.max-log-size=0 --viewer.make-share-url True --data /kaggle/temp/Cloudstudio/data/nerfstudio/poster --relative-model-dir /kaggle/working --output-dir /kaggle/working  --save-only-latest-checkpoint False --steps-per-save 2000 --logging.local-writer.stats-to-track ETA TOTAL_TRAIN_TIME CURR_TEST_PSNR",
        "local": False
    },
    "Devolver escena entrenada": {
        "command": "hypershell-copy token:/kaggle/working/checkpoint.ckpt /kaggle/temp/Cloudstudio/data/nerfstudio/poster",
        "local": True
    },
    "LERF": {
        "command": "uv pip install  --no-progress --python=$(which python) git+https://github.com/kerrj/lerf && ns-train lerf --logging.local-writer.max-log-size=0 --viewer.make-share-url True --data /kaggle/temp/Cloudstudio/data/nerfstudio/poster --relative-model-dir /kaggle/working --output-dir /kaggle/working  --save-only-latest-checkpoint False --steps-per-save 2000 --logging.local-writer.stats-to-track ETA TOTAL_TRAIN_TIME CURR_TEST_PSNR",
        "local": False
    },
    "Instruct-Nerf2Nerf": {
        "command": "uv pip install  --no-progress --python=$(which python) git+https://github.com/ayaanzhaque/instruct-nerf2nerf && ns-train in2n --logging.local-writer.max-log-size=0 --viewer.make-share-url True --data /kaggle/temp/Cloudstudio/data/nerfstudio/poster --relative-model-dir /kaggle/working --output-dir /kaggle/working  --save-only-latest-checkpoint False --steps-per-save 2000  --pipeline.prompt 'Replicate the poster in the TV' --logging.local-writer.stats-to-track ETA TOTAL_TRAIN_TIME CURR_TEST_PSNR",
        "local": False
    }  
}

def log_to_terminal(message):
    print(message)
    TERMINAL_QUEUE.put(message)

class Hypershell:
    def __init__(self):
        self.tunnel_process = None
        self.puerto = 5000
        self.api_url = "localhost"
        self.installed = True
        self.connected = False
        self.token = ""
        self.output_queue = queue.Queue()
        self.cpu_usage = 0
        self.ram_usage = 0
        self.last_metric_time = 0
        self.checkURL = True
        self.viser_url = ""
        self.connecting = False
        self.should_run = False   
        self.should_connect = False
        self.private_key = ""
        self.gpu_usage = 0
        self.gpu_memory = 0
        self.sio = Client()
        self.metrics = {}

    def run(self):
        if not self.installed:
            self.install()
        
        if self.private_key == "":
            self.private_key = self.load_keys()
        if not self.private_key:
            self.setup_keys()

        if self.should_connect:
            self.create_tunnel()
        
        @self.sio.on('connect')
        def on_connect():
            self.connected = True
            self.connecting = False
            log_to_terminal('Conectado al servidor')

        @self.sio.on('disconnect')
        def on_disconnect():
            self.connected = False
            log_to_terminal('Desconectado del servidor')

        @self.sio.on('connect_error')
        def on_connect_error(data):
            self.connected = False
            self.connecting = True

        @self.sio.on('metrics_update')
        def on_metrics_update(data):
            self.metrics = data
            self.connected = True
            self.connecting = False

        while self.should_run:
            time.sleep(UPDATE_INTERVAL)
            if not self.sio.connected:
                try:
                    self.sio.connect(f'http://{self.api_url}:{self.puerto}', wait_timeout=10, transports=['websocket'])
                except Exception as e:
                    if not hasattr(self, '_connection_error_logged'):
                        self.connected = False
                        self.connecting = True
                        log_to_terminal(f"Reconectando")
                        self._connection_error_logged = True
            if self.checkURL and self.connected:
                self.URL()

    def install(self):
        log_to_terminal("Instalando Hypershell...")
        try:
            npm_path = shutil.which('npm')
            if npm_path is None:
                raise FileNotFoundError("npm no encontrado")
            
            subprocess.check_call([npm_path, "install", "-g", "hypershell"], shell=True)
            log_to_terminal("Hypershell configurado. Listo para usar")
            self.installed = True
        except subprocess.SubprocessError as e:
            log_to_terminal(f"Error al instalar Hypershell: {e}")
            raise

    def create_tunnel(self):
        if self.token:
            command = f"hypershell {self.token} -L {self.puerto}:localhost:{self.puerto}"
            self.tunnel_process = subprocess.Popen(command, shell=True)
            log_to_terminal(f"Túnel Hypershell creado en puerto {self.puerto}")
            self.connecting = True
        else:
            log_to_terminal("No se pudo crear el túnel Hypershell")

    def close_tunnel(self):
        if self.tunnel_process:
            self.tunnel_process.kill()
            log_to_terminal("Túnel Hypershell cerrado")

    def run_command(self, command):
        try:
            with requests.post(f"http://{self.api_url}:{self.puerto}/run", json={'command': command}, stream=True, timeout=None) as response:
                if response.status_code == 409:
                    log_to_terminal("Ya hay un proceso en ejecución. Usa 'Interrumpir proceso' para detenerlo.")
                    return
                elif response.status_code != 200:
                    log_to_terminal(f"Error: {response.status_code}")
                    log_to_terminal(response.text)
                    return

                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        self.output_queue.put(chunk)
                    else:
                        time.sleep(0.1)  # Pequeña pausa si no hay datos
        except requests.exceptions.RequestException as e:
            log_to_terminal(f"Error de conexión: {str(e)}")

    def stop_current_process(self):
        try:
            response = requests.post(f"http://{self.api_url}:{self.puerto}/stop", timeout=TIMEOUT)
            if response.status_code == 200:
                log_to_terminal("Proceso interrumpido")
            elif response.status_code == 404:
                log_to_terminal("No hay proceso en ejecución para interrumpir")
            else:
                log_to_terminal(f"Error al interrumpir el proceso: {response.status_code}")
        except requests.exceptions.RequestException as e:
            log_to_terminal(f"Error al enviar la señal de interrupción: {str(e)}")
        finally:
            # Limpiar cualquier salida restante
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break

    def URL(self):
        try:
            terminal_output = ""
            while not self.output_queue.empty():
                try:
                    terminal_output += self.output_queue.get_nowait()
                except queue.Empty:
                    break
            for line in terminal_output.splitlines():
                if 'https://' in line and '.share.viser.studio' in line:
                    self.viser_url = line.strip()
                    self.checkURL = False
                    log_to_terminal(f"URL de Viser encontrada: {self.viser_url}")
                    break
            else:
                if not hasattr(self, '_viser_url_message_shown'):
                    log_to_terminal("Esperando a la URL de Viser...")
                    self._viser_url_message_shown = True
        except Exception as e:
            log_to_terminal(f"Error al leer la terminal: {e}. Reintentando...")

    def load_keys(self):
        if os.path.exists(f"{SCRIPT_DIR}/peer"):
            with open(f"{SCRIPT_DIR}/peer", 'r') as file:
                private_key = file.read().strip()
            with open(f"{SCRIPT_DIR}/config", "r") as file:
                self.token = file.read().strip()
            log_to_terminal(f"Configuracion cargada")
        else:
            private_key = None
            self.token = None
        return private_key

    def setup_keys(self):
        
            result = subprocess.run(f'hypershell-keygen -f {SCRIPT_DIR}/peer', capture_output=True, text=True, shell=True)
            log_to_terminal(f"Creando par de claves")
            if result.returncode != 0:
                log_to_terminal(f"Error al ejecutar el comando: {result.stderr}")
                return
            match = re.search(r"The public key is:\s*(\w+)", result.stdout)
            if not match:
                log_to_terminal("No se encontró la clave pública. Reinicia el programa")
                return
            self.token = match.group(1)
            with open(f"{SCRIPT_DIR}/config", "w") as file:
                file.write(self.token)
            #self.private_key = self.load_keys()
            self.restart()
            if not hasattr(self, '_token_logged'):
                log_to_terminal(f"Usando \"{self.token}\" como clave pública")
                self._token_logged = True

    def update_notebook_key(self):
        notebook_file = f"{SCRIPT_DIR}/tfgipy.ipynb"
        try:
            with open(notebook_file, 'r') as file:
                notebook_content = nbformat.read(file, as_version=4)
            if notebook_content.cells and notebook_content.cells[0].cell_type == 'code':
                first_cell = notebook_content.cells[0]
                pattern = r'echo\s"([a-zA-Z0-9]{52})"'
                first_cell.source = re.sub(pattern, f'echo "{self.private_key}"', first_cell.source)
            with open(notebook_file, 'w') as file:
                nbformat.write(notebook_content, file)
        except FileNotFoundError:
            log_to_terminal(f"Error: No se encontró el archivo {notebook_file}")
        except Exception as e:
            log_to_terminal(f"Error al procesar el archivo de la libreta: {str(e)}")

    def restart(self):
        self.stop()
        self.checkURL = True
        self.should_run = True
        self.create_tunnel()

    def stop(self):
        self.should_run = False
        self.close_tunnel()
        self.connected = False
        self.connecting = False
        
        if hasattr(self, 'tunnel_port'):
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(TIMEOUT)
            try:
                # Try to connect to the port
                sock.connect(('localhost', self.tunnel_port))
                # If successful, send a shutdown signal
                sock.shutdown(socket.SHUT_RDWR)
            except socket.error:
                # If connection fails, the port might already be closed
                pass
            finally:
                sock.close()
            
            # Try to forcefully close the port using psutil
            import psutil
            for conn in psutil.net_connections():
                if conn.laddr.port == self.tunnel_port:
                    try:
                        psutil.Process(conn.pid).terminate()
                    except psutil.NoSuchProcess:
                        pass
            
            delattr(self, 'tunnel_port')

        if self.sio.connected:
            self.sio.disconnect()

class CloudstudioGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.geometry("920x800")
        self.root.title("Cloudstudio")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.hypershell = Hypershell()
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()
        self.notebook_status = "Desconocido"
        self.kernel_slug = 'pablogranell/TFGipy'
        self.command_running = False
        self.active_notebook_states = ['running', 'starting', 'queued']
        self.status_map = {
            'running': 'Ejecutando',
            'queued': 'En cola',
            'starting': 'Iniciando',
            'cancelAcknowledged': 'Cancelado',
            'cancelRequested': 'Cancelando',
            'complete': 'Completado',
        }

        self.command_history = []
        self.history_index = -1

        self._run()
        self.input_field.focus_set()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _run(self):
        self._setup_ui()
        # Comprobacion inicial
        self._check_notebook_status()
        threading.Thread(target=self.hypershell.run, daemon=True).start()
        threading.Thread(target=self._update_status, daemon=True).start()
        threading.Thread(target=self._check_notebook_status, daemon=True).start()

    def _setup_ui(self):
        self._setup_json_section()
        self._setup_video_section()
        self._setup_viser_section()
        self._setup_notebook_section()
        self._setup_status_section()
        self._setup_terminal()
        self._setup_input_field()
        self._setup_predefined_commands()

    def _setup_json_section(self):
        json_frame = ctk.CTkFrame(self.root)
        json_frame.pack(pady=10, padx=10, fill='x')
    
        if self._check_kaggle_json():
            ctk.CTkLabel(json_frame, text="Archivo kaggle.json encontrado", text_color="green", font=("",20)).pack(side='left', padx=5)
        else:
            ctk.CTkLabel(json_frame, text="Archivo kaggle.json no encontrado", text_color="red", font=("",20)).pack(side='left', padx=5)
            ctk.CTkButton(json_frame, text="Buscar kaggle.json", command=self._browse_json).pack(side='left', padx=5)

    def _check_kaggle_json(self):
        if os.path.exists(KAGGLE_JSON_PATH):
            return True
        elif os.path.exists(os.path.join(SCRIPT_DIR, 'kaggle.json')):
            self._copy_json(os.path.join(SCRIPT_DIR, 'kaggle.json'))
            return True
        return False

    def _copy_json(self, src_path):
        try:
            os.makedirs(os.path.dirname(KAGGLE_JSON_PATH), exist_ok=True)
            shutil.copy(src_path, KAGGLE_JSON_PATH)
            log_to_terminal("Archivo kaggle.json copiado")
        except Exception as e:
            log_to_terminal(f"Error al copiar el archivo: {e}")

    def _browse_json(self):
        json_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if json_path.endswith('kaggle.json'):
            self._copy_json(json_path)
            self._setup_json_section()
        else:
            log_to_terminal("Archivo no válido, elige el archivo kaggle.json")
            
    def _setup_video_section(self):
        self.video_frame = ctk.CTkFrame(self.root)
        self.video_frame.pack(pady=10, padx=10, fill='x')
        self._update_video_status()
    
    def _update_video_status(self):
        for widget in self.video_frame.winfo_children():
            widget.destroy()
        video_exists = os.path.exists(os.path.join(SCRIPT_DIR, 'escena.mp4'))
        status_text = "Video encontrado" if video_exists else "Video no encontrado"
        status_color = "green" if video_exists else "red"
        ctk.CTkLabel(self.video_frame, text=status_text, text_color=status_color, font=("",20)).pack(side='left', padx=5)
        if not video_exists:
            ctk.CTkButton(self.video_frame, text="Buscar video", command=self._browse_video).pack(side='left', padx=5)

    def _browse_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if video_path.endswith('.mp4'):
            try:
                shutil.copy(video_path, os.path.join(SCRIPT_DIR, 'escena.mp4'))
                self._update_video_status()
            except Exception as e:
                log_to_terminal(f"Error al copiar el video: {e}")
        else:
            log_to_terminal("Archivo no válido, elige un archivo mp4")

    def _setup_viser_section(self):
        self.viser_frame = ctk.CTkFrame(self.root)
        self.viser_frame.pack(pady=10, padx=10, fill='x')
        self.viser_label = ctk.CTkLabel(self.viser_frame, text="Enlace Viser: No disponible", font=("",20))
        self.viser_label.pack(side='left', padx=5)
        self.viser_button = ctk.CTkButton(self.viser_frame, text="Abrir Viser",font=("",20), command=self._open_viser, state="disabled")
        self.viser_button.pack(side='left', padx=5)

    def _open_viser(self):
        if self.hypershell.viser_url:
            webbrowser.open(self.hypershell.viser_url)

    def _setup_notebook_section(self):
        notebook_frame = ctk.CTkFrame(self.root)
        notebook_frame.pack(pady=10, padx=10, fill='x')
        ctk.CTkButton(notebook_frame, text="Ejecutar Libreta Kaggle", font=("",20), 
                      command=self._run_kaggle_notebook).pack(side='left', padx=5)
        self.notebook_status_label = ctk.CTkLabel(notebook_frame, text=f"Estado de la libreta: {self.notebook_status}", font=("",20))
        self.notebook_status_label.pack(side='left', padx=5)
        
        token_frame = ctk.CTkFrame(self.root)
        token_frame.pack(pady=5, padx=10, fill='x')
        ctk.CTkButton(token_frame, text="Cambiar Token", width=190, font=("",20), 
                      command=self._change_token).pack(side='left', padx=10)
        self.token_entry = ctk.CTkEntry(token_frame, width=525, font=("",20), placeholder_text="Nuevo token")
        self.token_entry.pack(side='left', padx=(0, 10))

    def _setup_status_section(self):
        self.status_frame = ctk.CTkFrame(self.root)
        self.status_frame.pack(pady=10, padx=10, fill='x')
        
        self.connection_status = ctk.CTkLabel(self.status_frame, text="Conexion: Desconectado", text_color="red", font=("",20))
        self.connection_status.pack(side='left', padx=10)
        
        self.cpu_label = ctk.CTkLabel(self.status_frame, text="CPU: 0%", font=("",20))
        self.cpu_label.pack(side='left', padx=5)
        
        self.ram_label = ctk.CTkLabel(self.status_frame, text="RAM: 0%", font=("",20))
        self.ram_label.pack(side='left', padx=5)

        self.gpu_label = ctk.CTkLabel(self.status_frame, text="GPU: 0%", font=("",20))
        self.gpu_label.pack(side='left', padx=5)

        self.gpu_memory_label = ctk.CTkLabel(self.status_frame, text="GPU Mem: 0 MB", font=("",20))
        self.gpu_memory_label.pack(side='left', padx=5)
        
        self.prepared_status = ctk.CTkLabel(self.status_frame, text="Desconocido", text_color="red", font=("",20))
        self.prepared_status.pack(side='right', padx=10)

        # Nueva fila para el token
        self.token_frame = ctk.CTkFrame(self.root)
        self.token_frame.pack(pady=5, padx=10, fill='x')
        self.token_label = ctk.CTkLabel(self.token_frame, text="Token: ", font=("",20))
        self.token_label.pack(side='left', padx=10)

    def _setup_terminal(self):
        self.terminal_text = ctk.CTkTextbox(self.root, width=600, height=250, font=("", 25), state="disabled")
        self.terminal_text.pack(pady=10, padx=10, fill='both', expand=True)
        self.root.after(100, self._update_terminal)

    def _setup_input_field(self):
        self.input_frame = ctk.CTkFrame(self.root)
        self.input_frame.pack(pady=10, padx=10, fill='x')
        
        # Frame para la entrada, el botón de enviar y la barra de progreso
        self.input_send_frame = ctk.CTkFrame(self.input_frame)
        self.input_send_frame.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        self.input_field = ctk.CTkEntry(self.input_send_frame, height=40, font=("",25), placeholder_text="Escribe un comando")
        self.input_field.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.input_field.bind("<Return>", lambda event: self._send_input())
        self.input_field.bind("<Up>", self._previous_command)
        self.input_field.bind("<Down>", self._next_command)
        
        self.send_button = ctk.CTkButton(self.input_send_frame, text="Enviar", font=("",30), command=self._send_input)
        self.send_button.pack(side='left')
        
        # Crear el widget de progreso sobre el campo de entrada y el botón de enviar
        self.progress = ctk.CTkProgressBar(self.input_send_frame, orientation="horizontal", mode="indeterminate", height=40)
        self.progress.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.progress.set(0)
        self.progress.lower()  # Colocar la barra de progreso detrás de los otros widgets
        
        # Frame para el botón de interrupción
        self.interrupt_frame = ctk.CTkFrame(self.input_frame)
        self.interrupt_frame.pack(side='left', fill='y')
        
        self.interrupt_button = ctk.CTkButton(self.interrupt_frame, text="Interrumpir proceso", font=("",30), command=self.hypershell.stop_current_process)
        self.interrupt_button.pack(fill='both', expand=True)

    def _setup_predefined_commands(self):
        commands_frame = ctk.CTkFrame(self.root)
        commands_frame.pack(pady=10, padx=10, fill='x')
        for i, (command_name, command) in enumerate(PREDEFINED_COMMANDS.items()):
            ctk.CTkButton(commands_frame, text=command_name, font=("",20), 
                          command=lambda cmd=command: self._execute_predefined_command(cmd)).grid(row=i//3, column=i%3, padx=5, pady=5)

    def _execute_predefined_command(self, command):
        if not self.command_running:
            self.input_field.insert(0, command["command"])
            self._send_input(local=command["local"])

    def _on_closing(self):
        self.hypershell.stop()
        self.root.destroy()

    def _update_terminal(self):
        messages = []
        try:
            while True:
                if not TERMINAL_QUEUE.empty():
                    messages.append(TERMINAL_QUEUE.get_nowait())
                elif not self.hypershell.output_queue.empty() and not self.hypershell.checkURL:
                    messages.append(self.hypershell.output_queue.get_nowait())
                else:
                    break
        except queue.Empty:
            pass

        if messages:
            cleaned_messages = [self._clean_terminal_output(msg) for msg in messages if msg]
            if cleaned_messages:
                self.terminal_text.configure(state="normal")  # Temporalmente habilitar la edición
                self.terminal_text.insert("end", "\n".join(cleaned_messages) + "\n")
                self.terminal_text.see("end")
                self.terminal_text.configure(state="disabled")  # Volver a deshabilitar la edición
                
        self.root.after(UPDATE_INTERVAL, self._update_terminal)
        #self.input_field.focus_set()

    def _clean_terminal_output(self, text):
        
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])') # Eliminar códigos de escape ANSI y líneas vacías
        cleaned_text = ansi_escape.sub('', text) # Eliminar líneas vacías consecutivas
        cleaned_text = '\n'.join(line for line in cleaned_text.splitlines() if line.strip()) # Eliminar múltiples saltos de línea consecutivos, dejando máximo dos
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        return cleaned_text

    def _send_input(self, local=False):
        user_input = self.input_field.get()
        if user_input:
            self.command_history.insert(0, user_input)
            self.history_index = -1
            self.command_running = True
            self.input_field.delete(0, 'end')
            self.input_field.configure(state="disabled")  # Deshabilitar la entrada
            self.send_button.configure(state="disabled")  # Deshabilitar el botón de enviar
            self.progress.lift()  # Traer la barra de progreso al frente
            self.progress.start()  # Iniciar la animación
            threading.Thread(target=self._run_command, args=(user_input, local)).start()

    def _run_command(self, command, local=False):
        try:
            if local:
                log_to_terminal(f"Ejecutando comando local")
                self._run_local_command(command)
            else:
                log_to_terminal(f"Comando enviado. Espera hasta un minuto")
                self.hypershell.run_command(command)
        finally:
            self.root.after(0, self._command_finished)

    def _run_local_command(self, command):
        try:
            self.command_running = True
            # Reemplazar 'token' con el token actual de Hypershell
            command = command.replace('token', self.hypershell.token)
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            log_to_terminal(f"Comando ejecutado")
        except subprocess.CalledProcessError as e:
            self.command_running = False
            log_to_terminal(f"Error al ejecutar el comando local:\n{e.stderr}")

    def _command_finished(self):
        self.command_running = False
        self.progress.stop()  # Detener la animación
        self.progress.lower()  # Enviar la barra de progreso al fondo
        self.input_field.configure(state="normal")  # Habilitar la entrada
        self.send_button.configure(state="normal")  # Habilitar el botón de enviar
        self.input_field.focus_set()

    def _run_kaggle_notebook(self):
        if self.notebook_status in ['Ejecutando', 'En cola', 'Iniciando']:
            log_to_terminal("Ya hay una libreta en ejecución. Por favor, espera a que termine.")
            self.hypershell.should_run = True
            return
        threading.Thread(target=self._execute_kaggle_notebook, daemon=True).start()
        self.hypershell.should_run = True
        self.hypershell.should_connect = True  # Activar la conexión después de iniciar la libreta

    def _execute_kaggle_notebook(self):
        try:
            log_to_terminal("Ejecutando la libreta Kaggle, esto puede tardar hasta 4 minutos")
            self.notebook_status = "Iniciando"
            self.notebook_status_label.configure(text=f"Estado de la libreta: {self.notebook_status}")
            self.kaggle_api.kernels_pull(kernel=self.kernel_slug, path=SCRIPT_DIR, metadata=True)
            self.hypershell.update_notebook_key()
            self.kaggle_api.kernels_push(SCRIPT_DIR)
            time.sleep(2)
            threading.Thread(target=self.hypershell.run, daemon=True).start()
        except Exception as e:
            log_to_terminal(f"Error al ejecutar la libreta Kaggle: {e}")
            self.notebook_status = "Error"
            self.notebook_status_label.configure(text=f"Estado de la libreta: {self.notebook_status}")
            self.hypershell.should_connect = False

    def _change_token(self):
        new_token = self.token_entry.get().strip()
        if new_token:
            self.hypershell.token = new_token
            log_to_terminal(f"Token cambiado a: {new_token}")
            log_to_terminal("Reiniciando Hypershell con el nuevo token...")
            self.hypershell.restart()
            self.token_entry.delete(0, 'end')
        else:
            log_to_terminal("Error: El token no puede estar vacío")

    def _update_status(self):
        while True:
            self._update_connection_status()
            self._update_token_and_viser()
            self._check_notebook_status()
            time.sleep(UPDATE_INTERVAL)

    def _check_notebook_status(self):
        try:
            status = self.kaggle_api.kernels_status(self.kernel_slug)['status']
            self.notebook_status_label.configure(text=f"Estado de la libreta: {self.status_map.get(status, status)}")
            self.hypershell.should_connect = self.hypershell.should_run = status in self.active_notebook_states
        except Exception as e:
            log_to_terminal(f"Error al obtener el estado de la libreta: {e}")

    def _update_connection_status(self):
        if self.hypershell.connecting:
            status, color = "Conectando", "orange"
        elif self.hypershell.connected:
            status, color = "Conectado", "green"
        else:
            status, color = "Desconectado", "red"
        self.connection_status.configure(text=f"Conexion: {status}", text_color=color)
        
        prepared_states = {(False, False): ("Desconocido", "red"),(True, True): ("Ejecutando", "orange"),(True, False): ("Preparado", "green"), (False, True): ("En cola", "orange")}
        status1, color1 = prepared_states[(self.hypershell.connected, self.command_running)]
        self.prepared_status.configure(text=status1, text_color=color1)
        self.cpu_label.configure(text=f"CPU: {self.hypershell.metrics.get('cpu', 0):.1f}%")
        self.ram_label.configure(text=f"RAM: {self.hypershell.metrics.get('memory', 0):.1f}%")
        self.gpu_label.configure(text=f"GPU: {self.hypershell.metrics.get('gpu_load', 0):.1f}%")
        self.gpu_memory_label.configure(text=f"GPU Mem: {self.hypershell.metrics.get('gpu_memory', 0):.1f} MB")
        
    def _update_token_and_viser(self):
        if self.hypershell.checkURL:
            self.viser_label.configure(text="Enlace Viser: No disponible")
        else:
            self.hypershell.viser_url = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', self.hypershell.viser_url).strip()
            self.viser_label.configure(text=f"Enlace Viser: {self.hypershell.viser_url}")
            self.viser_button.configure(state="normal")
        self.token_label.configure(text=f"Token: {self.hypershell.token}")
    
    def _previous_command(self, event):
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.input_field.delete(0, 'end')
            self.input_field.insert(0, self.command_history[self.history_index])
        return "break"

    def _next_command(self, event):
        if self.history_index > 0:
            self.history_index -= 1
            self.input_field.delete(0, 'end')
            self.input_field.insert(0, self.command_history[self.history_index])
        elif self.history_index == 0:
            self.history_index = -1
            self.input_field.delete(0, 'end')
        return "break"

if __name__ == "__main__":
    CloudstudioGUI()
import os
from argparse import Namespace
from fastapi import FastAPI
from PIL import Image
import numpy as np

from io import BytesIO
from argparse import Namespace
import torch
import threading
import time
import psutil
import sys
from plugin import Plugin, fetch_pil_image, store_pil_image, store_multiple, fetch_multiple, fetch_image, store_image
from .config import plugin, config, endpoints

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.utils.img_util import tensor2img
from basicsr.utils import img2tensor
from basicsr.archs.swinir_arch import SwinIR
from .inference.inference_swinir import define_model
from torch.nn import functional as F





app = FastAPI()

def check_model():
    if 'sr_plugin' not in globals():
        set_model()

@app.get("/get_info/")
def plugin_info():
    check_model()
    return sr_plugin.plugin_info()

@app.get("/get_config/")
def get_config():
    check_model()
    return sr_plugin.get_config()

@app.put("/set_config/")
def set_config(update: dict):
    sr_plugin.set_config(update) # TODO: Validate config dict are all valid keys
    return sr_plugin.get_config()

@app.on_event("startup")
async def startup_event():
    print("Starting up")
    # A slight delay to ensure the app has started up.
    try:
        set_model()
        print("Successfully started up")
        sr_plugin.notify_main_system_of_startup("True")
    except Exception as e:
        # raise e
        sr_plugin.notify_main_system_of_startup("False")

@app.get("/set_model/")
def set_model():
    global sr_plugin
    args = {"plugin": plugin, "config": config, "endpoints": endpoints}
    sr_plugin = SR(Namespace(**args))
    # try:
    # sd_plugin.set_model(args["model_name"], dtype=args["model_dtype"])
    # model_name = sr_plugin.config["model_name"
    return {"status": "Success", "detail": f"Model set successfully"}

@app.get("/execute/{img_id}")
def execute(img_id: str):
    # check_model()

    image = fetch_pil_image(img_id)
    # image = Image.open(BytesIO(imagebytes))

    output_img = sr_plugin.super_res(image)
    # output = BytesIO()

    if output_img is None:
        return {"status": "Failed", "detail": "Super resolution failed"}
    
    # output_img.save(output, format="PNG")
    output_img_id = store_pil_image(output_img)

    return {"status": "Success", "output_id": output_img_id}

@app.get("/video_superres/{img_list_id}")
def video_superres(img_list_id: str):
    # check_model()
    id_data = fetch_image(img_list_id)
    id_list = id_data.decode("utf-8").split(";")
    print(id_list, len(id_list))
    # frames = fetch_multiple_images(id_list)
    # pil_frames = [fetch_pil_image(id) for id in id_list]
    pil_frames = fetch_multiple(fetch_pil_image, id_list)
    image_list = [np.array(frame) for frame in pil_frames]
    image_list = np.array(image_list)
    # image = np.array(image)
    output_list = sr_plugin.video_super_res(image_list)
    pil_output = [Image.fromarray(frame) for frame in output_list]
    # image_ids = [store_pil_image(frame) for frame in pil_output]
    # image_id = store_image(bytes(";".join(image_ids).encode("utf-8")))
    image_id = store_multiple(pil_output, store_pil_image)

    return {"status": "Success", "output_id": image_id}

def self_terminate():
    time.sleep(3)
    parent = psutil.Process(psutil.Process(os.getpid()).ppid())
    print(f"Killing parent process {parent.pid}")
    # os.kill(parent.pid, 1)
    # parent.kill()

@app.get("/shutdown/")  #Shutdown the plugin
def shutdown():
    threading.Thread(target=self_terminate, daemon=True).start()
    return {"success": True}

class SR(Plugin):
    """
    Prediction inference.
    """
    def __init__(self, arguments: "Namespace") -> None:
        super().__init__(arguments)
        self.plugin_name = "BasicSR"
        model_folder = "plugin/BasicSR/experiments/pretrained_models/"
        self.esrgan_model_path = os.path.join(model_folder, arguments.config["esrgan_model"])
        self.swinir_model_path = os.path.join(model_folder, arguments.config["swinir_model"])
        if sys.platform == "darwin":
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_model()


    def load_model(self, model_path: str, model: torch.nn.Module) -> None:
        model.load_state_dict(torch.load(model_path)['params'], strict=True)
        model.eval()
        model = model.to(self.device)

    def set_model(self) -> None:
        """
        Load given weights into model.
        """
        # Load ESRGAN
        if self.esrgan_model_path is not None:
            self.esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            self.load_model(self.esrgan_model_path, self.esrgan_model)

        # Load SwinIR
        if self.swinir_model_path is not None:
            target_model = self.swinir_model_path.split("/")[-1]
            split_name = target_model.split("_")
            task, scale, patch_size = split_name[1], int(split_name[-1].split("x")[1].split(".")[0]), int(split_name[3][1:3])
            if task == "classicalSR":
                task = "classical_sr"
            swin_args = {"task": task, "scale": scale, "patch_size": patch_size, "model_path": self.swinir_model_path}
            swin_input = Namespace(**swin_args)
            self.swin_model = define_model(swin_input)
            self.swin_scale = scale
            self.load_model(self.swinir_model_path, self.swin_model)

        self.vsr_model = BasicVSR(num_feat=64, num_block=30)
        self.vsr_model.to(self.device)
        self.interval = 12
        self.save_path = "plugin/BasicSuperRes/results/BasicVSR"
    

    def super_res(self, inputs, model="esrgan"):
        """
        Super resolution inference.
        """
        image = inputs
        image = np.array(image) / 255
        img = torch.from_numpy(np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        if model == "esrgan":
            with torch.no_grad():
                esrgan_output = self.esrgan_model(img)
                output_img = self.to_image(esrgan_output)
        elif model == "swinir":
            if self.swin_model is not None:
                with torch.no_grad():
                    window_size = 8
                # pad input image to be a multiple of window_size
                    mod_pad_h, mod_pad_w = 0, 0
                    _, _, h, w = img.size()
                    if h % window_size != 0:
                        mod_pad_h = window_size - h % window_size
                    if w % window_size != 0:
                        mod_pad_w = window_size - w % window_size
                    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                    output = self.swin_model(img)
                    _, _, h, w = output.size()
                    swinir_output = output[:, :, 0:h - mod_pad_h * self.swin_scale, 0:w - mod_pad_w * self.swin_scale]
            output_img = self.to_image(swinir_output)
        return output_img

    def to_image(self, input):
        if input is not None:
            output = input.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            output = Image.fromarray(output)
            return output
        else:
            return None

    def video_super_res(self, image_list):
        new_img_list = []
        num_imgs = len(image_list)
        if len(image_list) <= self.interval:  # too many images may cause CUDA out of memory
            imgs = read_img_seq(image_list)
            imgs = imgs.unsqueeze(0).to(self.device)
            result = self.basicvsr_inference(imgs, self.vsr_model, self.save_path)
            new_img_list.extend(result)
        else:
            for idx in range(0, num_imgs, self.interval):
                interval = min(self.interval, num_imgs - idx)
                imgs = image_list[idx:idx + interval]
                imgs = read_img_seq(imgs)
                imgs = imgs.unsqueeze(0).to(self.device)
                result = self.basicvsr_inference(imgs, self.vsr_model, self.save_path)
                new_img_list.extend(result)
        return new_img_list

    
    def basicvsr_inference(self, imgs, model, save_path):
        output_list = []
        with torch.no_grad():
            outputs = model(imgs)
        # save imgs
        outputs = outputs.squeeze()
        outputs = list(outputs)
        for output in outputs:
            output = tensor2img(output)
            output_list.append(output)
        return output_list
    
def read_img_seq(imgs):
    imgs = [img.astype(np.float32) / 255. for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)
    return imgs
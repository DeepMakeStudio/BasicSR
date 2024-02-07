import sys

plugin = {
    "Name": "Super Resolution (BasicSR)",
    "Version": "0.1.0", 
    "Author": "DeepMake",
    "Description": "Super Resolution using BasicSR",
    "env": "basicsr"
}
config = {
    "esrgan_model": "ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    "swinir_model": "SwinIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
    "model_dtype": "fp32" if sys.platform == "darwin" else "fp16"
}
endpoints = {
    "upres_image": {
        "call": "execute",
        "inputs": {
            "img": "Image",
        },
        "outputs": {"output_img": "Image"}
    },
    "upres_video": {
        "call": "video_superres",
        "inputs": {
            "img_list_id": "ImageList",
        },
        "outputs": {"output_imgs": "Image"}
    }
}
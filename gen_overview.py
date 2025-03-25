import os
from glob import glob
from PIL import Image
import numpy as np


template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Gaussian Image Upscaling</title>
    <meta name="description" content="Comparison of different upscaling methods">
    <meta name="author" content="Simon Niedermayr">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/normalize.css@8/normalize.css">
    <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="header">
    <h1>Images Encoded as Gaussian Mixture Models</h1>
    <h2>Comparison of different upscaling methods</h2>
    <p>
    2D gaussian mixture model was fitted to images with a similar method as proposed by <a href="https://github.com/Xinjie-Q/GaussianImage">GaussianImage</a>.
    The decoding or rendering resolution can be changed dynamically and upscaling methods can be compared. 
    Our method spline utilizes analytical gradients and is able to upscale images with higher quality.
    The images are decoded / rendered in real time with WebGPU.
    </p>
</div>
{body}
</div>
</body>
"""

body = ""

def gen_dataset_thumbnails(name,description,link,root_dir):

    body= f"""
    <div class="dataset">
        <a href={link}>
        <h2>{name}</h2>
        </a>
        <p>{description}</p>
    </div>
    """
    body +="""<div class="container">"""
    for folder in sorted(os.listdir(root_dir)):
        try:
            file_dir = os.path.join(root_dir,folder)
            if not os.path.isdir(file_dir):
                continue
            img_name = os.path.basename(glob(f"{file_dir}/*.png")[0])
            img = Image.open(os.path.join(file_dir,img_name))
            
            orig_resolution = f"{img.width}x{img.height}px"
            ratio = img.width/img.height
            img.resize((640,int(640/ratio))).save(os.path.join(file_dir,"thumbnail.jpeg"))
            model_file = "model.npz"

            model = np.load(os.path.join(file_dir,model_file))
            n_gaussians = model["xyz"].shape[0]

            def tof16(x):
                if x.dtype.kind == 'f':
                    return x.astype(np.float16)
                return x

            model_f16 = {k: tof16(model[k]) for k in model.files}
            np.savez_compressed(os.path.join(file_dir,"model_f16.npz"),**model_f16)

            
            size = os.path.getsize(os.path.join(file_dir, "model_f16.npz"))
            if size < 1024 * 1024:
                size = f"{size // 1024:.2f} KB"
            else:
                size = f"{size / (1024 * 1024):.2f} MB"

            base_dir = os.path.relpath(file_dir, file_dir.split("/")[0])

            body += f"""
            <div class="card">
                <p class="name">{folder}</p>
                <a href="viewer.html?file={os.path.join(base_dir,"model_f16.npz")}">
                    <div class="image" style="background-image: url({os.path.join(base_dir,'thumbnail.jpeg')})"></div>
                </a>
                <p class="info">{size}<br> {n_gaussians//1000}k Gauss.<br>{orig_resolution}</p>
            </div>
            """
        except Exception as e:
            print(folder,e)


    body +="""</div>"""
    return body

body += gen_dataset_thumbnails("DIV2K","2K resolution high quality images","https://data.vision.ee.ethz.ch/cvl/DIV2K/","public/gaussian_images/div2k")
body += """<hr>"""
body += gen_dataset_thumbnails("Kodak","Lossless True Color Image Suite","https://r0k.us/graphics/kodak/","public/gaussian_images/kodak")


with open("public/index.html", "w") as f:
    f.write(template.format(body=body))
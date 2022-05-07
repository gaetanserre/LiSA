from PIL import Image
import os

images_path = []
for file in os.listdir("."):
  name, ext = os.path.splitext(file)
  if ext == ".ppm":
    im = Image.open(file)
    im.save(name + ".png", format="png")
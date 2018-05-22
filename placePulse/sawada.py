from io import BytesIO
from PIL import Image
import streetview
import requests


def api_download2(panoid, ppid, heading, flat_dir, key, width=640, height=640,
                  fov=120, pitch=0, extension='jpg', year=2017):
    """
    Download an image using the official API. These are not panoramas.
    Params:
        :panoid: the panorama id
        :heading: the heading of the photo. Each photo is taken with a 360
            camera. You need to specify a direction in degrees as the photo
            will only cover a partial region of the panorama. The recommended
            headings to use are 0, 90, 180, or 270.
        :flat_dir: the direction to save the image to.
        :key: your API key.
        :width: downloaded image width (max 640 for non-premium downloads).
        :height: downloaded image height (max 640 for non-premium downloads).
        :fov: image field-of-view.
        :image_format: desired image format.
    You can find instructions to obtain an API key here: https://developers.google.com/maps/documentation/streetview/
    """

    fname = str(ppid)  # "%s_%s_%s_%s" % (year, panoid,str(ppid),str(heading))
    image_format = extension if extension != 'jpg' else 'jpeg'

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        # maximum permitted size for free calls
        "size": "%dx%d" % (width, height),
        "fov": fov,
        "pitch": pitch,
        "heading": heading,
        "pano": panoid,
        "key": key
    }

    response = requests.get(url, params=params, stream=True)
    try:
        img = Image.open(BytesIO(response.content))
        filename = '%s/%s.%s' % (flat_dir, fname, extension)
        img.save(filename, image_format)
    except:
        print("Image not found")
        filename = None
    del response
    return filename

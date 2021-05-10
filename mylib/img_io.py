import cv2
import numpy as np
import OpenEXR, Imath

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

        	
def load_image(input_dir):
	return cv2.imread(input_dir)

    
def load_image_npy(input_dir):
	return np.load(input_dir)	



def writeLDR(img, file, exposure=1.0): 
	sc = np.power(np.power(2.0, exposure), 0.5)
	img = ((sc * img) * 255).astype(np.uint8) 
	try:
		cv2.imwrite(file, img)
	except Exception as e:
		raise IOException("Failed writing LDR image: %s"%e)
    

def writeNPY(image, image_name):
	try:
		np.save(image_name, image)
	except Exception as e:
		raise IOException("Failed writing HDR image: %s"%e)


def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        imgR = img[:,:,2]
        imgG = img[:,:,1]
        imgB = img[:,:,0]
        img = np.dstack((imgR, imgG, imgB))
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        R = (img[:,:,0]).astype(np.float32).tostring()
        G = (img[:,:,1]).astype(np.float32).tostring()
        B = (img[:,:,2]).astype(np.float32).tostring()
        out.writePixels({'R' : R, 'G' : G, 'B' : B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)


def loadEXR(name_hdr):
    return pyexr.read_all(name_hdr)['default'][:,:,0:3]
    


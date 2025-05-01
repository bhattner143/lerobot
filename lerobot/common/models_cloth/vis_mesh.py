import cv2 as cv
import numpy as np

#  paint circle within image
def paint_circle(image, center, radius=4, color=(0, 255, 0), size=-1):
    cv.circle(image, (int(center[1]), int(center[0])), radius, color, size)
    return image

# paint rectangle within image
def paint_rectangle(image, center, dh, dw, color=(0, 255, 0), size=-1):
    cv.rectangle(image, (int(center[1]-dw), int(center[0]-dh)), (int(center[1]+dw), int(center[0]+dh)), color, size)
    return image

# paint tritangle within image
def paint_triangle(image, center, radius=10, color=(0, 255, 0), size=-1):
    p0 = [center[0]-radius/2, center[1]-radius*(3**0.5)/2]
    p1 = [center[0]-radius/2, center[1]+radius*(3**0.5)/2]
    p2 = [center[0]+radius, center[1]]
    paint_line(image, p0, p1, color=color, size=size)
    paint_line(image, p1, p2, color=color, size=size)
    paint_line(image, p2, p0, color=color, size=size)
    return image

# paint line within image
def paint_line(image, start, end, color=(0, 255, 0), size=10):
    cv.line(image, (int(start[1]), int(start[0])), (int(end[1]), int(end[0])), color, size)
    return image

if __name__ == "__main__":
    # load the mesh_0.txt as a numpy array
    mesh = np.loadtxt("mesh_0.txt", dtype=np.float32)

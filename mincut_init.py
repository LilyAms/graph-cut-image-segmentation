import numpy as np
import imageio
import matplotlib.pyplot as plt
import maxflow as gc
from copy import copy

from skimage import data, img_as_float

def binary_restore(I, Lambda=50):
    """I: noisy image to be restored.
    Lambda: weight of regularization factor.
    Return the denoised binary image."""
    g = gc.Graph[int]()
    nodeids = g.add_grid_nodes(I.shape)
    g.add_grid_edges(nodeids, Lambda)
    g.add_grid_tedges(nodeids, I, 255-I)
    g.maxflow()
    # Get the source/sink label
    labels = g.get_grid_segments(nodeids)
    I2 = np.int_(np.logical_not(labels))
    return I2

# Binary image restoration
I=imageio.imread('binary_image2.png')[:,:,0]
I2 = np.clip(I+np.random.normal(0, 100, I.shape),0,255).astype(np.uint8)
plt.subplot(131)
plt.imshow(I,cmap='gray')
plt.subplot(132)
plt.imshow(I2,cmap='gray')
plt.subplot(133)
plt.imshow(binary_restore(I2,70),cmap='gray')
plt.show()

# ------------------------------------------------- #

class Scribbler:
    """The scribbler lets you draw red and green zones with left and right mouse
    buttons"""
    def __init__(self, im):
        self.im = im
        self.channel = None
        self.cidpress = self.im.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.im.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.im.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def display(self,event):
        x,y = int(event.xdata),int(event.ydata)
        self.im.get_array()[y-2:y+3,x-2:x+3, self.channel]      = 255
        self.im.get_array()[y-2:y+3,x-2:x+3,(self.channel+1)%3] = 0
        self.im.get_array()[y-2:y+3,x-2:x+3,(self.channel+2)%3] = 0
        self.im.figure.canvas.draw()

    def on_press(self, event):
        if event.button==1:
            self.channel = 1
        else:
            self.channel = 0
        self.display(event)

    def on_motion(self, event):
        if self.channel is None: return
        self.display(event)

    def on_release(self, event):
        self.channel = None

def compute_probabilities(I,scribbledI):
    
    #Extract pixels marked green and red by hand with the scribbler
    green_pixels=np.all(scribbledI[:,:]==(0,255,0),axis=2)
    red_pixels=np.all(scribbledI[:,:]==(255,0,0),axis=2)
   
    #build histogram
    intensities=np.arange(257)
    P0=np.histogram(I[red_pixels],bins=intensities,density=True)[0]
    P1=np.histogram(I[green_pixels],bins=intensities,density=True)[0]

    return (P0,P1)

def minCut(I,k=1e-5,Lambda=10,sigma=0.5,strong_edges=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im=ax.imshow(np.stack((I,)*3, axis=-1))
    scribbledI=Scribbler(im)
    Irg=im.get_array()
    plt.show()
    
    P0,P1=compute_probabilities(I,Irg)
    
    g = gc.Graph[int]()
    nodeids = g.add_grid_nodes(I.shape)
    
    #Avoid penalizing cuts along strong edges
    #Structure used for the graph is "Von Neumann", 4 neighbouring pixels (up, down, right and left)
    if (strong_edges):
        structure=np.zeros((3,3))
        
        #Down neighbouring pixel
        structure[2,1]=1
        weights=Lambda*np.exp(-(I-np.roll(I,-1,axis=0))**2/sigma**2)
        g.add_grid_edges(nodeids,weights,structure,symmetric=False)
        
        #Up neighbouring pixel
        structure[2,1]=0
        structure[0,1]=1
        weights=Lambda*np.exp(-(I-np.roll(I,1,axis=0))**2/sigma**2)
        g.add_grid_edges(nodeids,weights,structure,symmetric=False)
        
        #Left neighbouring pixel
        structure[0,1]=0
        structure[1,0]=1
        weights=Lambda*np.exp(-(I-np.roll(I,1,axis=1))**2/sigma**2)
        g.add_grid_edges(nodeids,weights,structure,symmetric=False)
        
        #Right neighbouring pixel
        structure[1,2]=0
        structure[1,2]=1
        weights=Lambda*np.exp(-(I-np.roll(I,-1,axis=1))**2/sigma**2)
        g.add_grid_edges(nodeids,weights,structure,symmetric=False)
    
    else:
        g.add_grid_edges(nodeids, Lambda)
    
    g.add_grid_tedges(nodeids, -np.log(k+P0[I]), -np.log(k+P1[I]))
    g.maxflow()
    # Get the source/sink label
    labels = g.get_grid_segments(nodeids)
    I3 = np.int_(np.logical_not(labels))
    
    plt.subplot(121)
    plt.imshow(im.get_array())
    plt.subplot(122)
    plt.imshow(I3)
    plt.show()

    
    

I=imageio.imread('coins.png').astype(np.uint8)
minCut(I,strong_edges=True)



# Demo: small Python application demonstrating pixel, texel, voxel, surfel, spaxel, phoxel, toxel, flexel, and voxtexel
# This single script synthesizes toy data for each concept and visualizes them.
# It follows the rule: use matplotlib (no seaborn), keep each plot separate.
# Run in a Jupyter environment. Outputs (images and plots) will be displayed below.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# ------------------ 1) Pixel: simple 2D image ------------------
def demo_pixels():
    # create a checkerboard 2D image (pixels)
    img = np.indices((128,128)).sum(axis=0) % 2
    plt.figure(figsize=(4,4))
    plt.title("Pixels: Checkerboard (128x128)")
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')

# ------------------ 2) Texel: texture element ------------------
def demo_texels():
    # create a small texture (texels) and map onto a rectangle
    tex = np.zeros((32,32,3))
    for i in range(32):
        for j in range(32):
            tex[i,j] = [(i/31), (j/31), ((i+j)/62)]
    # show texture
    plt.figure(figsize=(3,3))
    plt.title("Texels: 32x32 texture")
    plt.imshow(tex, interpolation='nearest')
    plt.axis('off')

    # "Map" texture onto a coarse grid surface (UV coordinates)
    u = np.linspace(0,1,8)
    v = np.linspace(0,1,8)
    U, V = np.meshgrid(u,v)
    mapped = np.zeros((U.shape[0], U.shape[1], 3))
    mapped[:,:,0] = U
    mapped[:,:,1] = V
    mapped[:,:,2] = 0.5*(U+V)
    plt.figure(figsize=(4,3))
    plt.title("Texels mapped to patch (coarse UV)")
    plt.imshow(mapped, origin='lower', interpolation='nearest')
    plt.axis('off')

# ------------------ 3) Voxel: 3D volume element ------------------
def demo_voxels():
    # create a 3D gaussian sphere volume
    size = 32
    x = np.linspace(-1,1,size)
    X,Y,Z = np.meshgrid(x,x,x)
    vol = np.exp(-((X**2+Y**2+Z**2)/(2*0.2**2)))
    # show a central slice
    plt.figure(figsize=(4,4))
    plt.title("Voxels: central slice of 3D volume")
    plt.imshow(vol[:,:,size//2], origin='lower')
    plt.axis('off')
    # 3D scatter of thresholded voxels (sparse visualization)
    thresh = 0.2
    pts = np.argwhere(vol>thresh)
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Voxels (sparse scatter)")
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=4)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

# ------------------ 4) Surfel: surface element ------------------
def demo_surfels():
    # sample points on a sphere surface and compute normals (surfels)
    n = 500
    phi = np.random.uniform(0,2*np.pi,n)
    costheta = np.random.uniform(-1,1,n)
    theta = np.arccos(costheta)
    r = 1.0
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    pts = np.vstack([x,y,z]).T
    normals = pts.copy()  # for a sphere, normal ~ position
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Surfels: point samples + normals (sphere)")
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=6)
    # show a small subset of normals using quiver
    subset = np.random.choice(n, size=40, replace=False)
    ax.quiver(pts[subset,0], pts[subset,1], pts[subset,2],
              normals[subset,0]*0.2, normals[subset,1]*0.2, normals[subset,2]*0.2, length=0.2)
    ax.set_box_aspect((1,1,1))

# ------------------ 5) Spaxel: spatial pixel with spectrum ------------------
def demo_spaxels():
    # create an image where each pixel stores a tiny spectrum (e.g., 10 wavelengths)
    H,W = 32,32
    wavelengths = np.linspace(400,700,10)  # nm
    spectra = np.zeros((H,W,len(wavelengths)))
    # assign spectra that vary with x,y (toy model)
    for i in range(H):
        for j in range(W):
            center = 500 + 150*(i/H - 0.5) + 100*(j/W - 0.5)
            spectra[i,j] = np.exp(-0.5*((wavelengths - center)/20)**2)
    # show RGB projection for visual reference (simply sum bands into 3 channels)
    rgb = np.zeros((H,W,3))
    rgb[:,:,0] = spectra[:,:,2].copy()  # blue-ish band
    rgb[:,:,1] = spectra[:,:,5].copy()  # green-ish band
    rgb[:,:,2] = spectra[:,:,8].copy()  # red-ish band
    plt.figure(figsize=(4,4))
    plt.title("Spaxels: RGB projection of spectral image")
    plt.imshow(rgb, interpolation='nearest')
    plt.axis('off')
    # pick a pixel and plot its spectrum
    pi,pj = H//2, W//2
    plt.figure(figsize=(5,2.5))
    plt.title(f"Spaxel spectrum at pixel ({pi},{pj})")
    plt.plot(wavelengths, spectra[pi,pj])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative intensity")

# ------------------ 6) Phoxel: photon voxel (toy photon mapping) ------------------
def demo_phoxels():
    # simulate N photons emitted from a point; do a tiny random-walk scatter and accumulate in 3D grid
    size = 32
    grid = np.zeros((size,size,size))
    origin = np.array([size//2, size//2, size//2])
    N = 4000
    for _ in range(N):
        pos = origin.copy().astype(float)
        for step in range(20):  # limited steps
            # small random step
            stepvec = np.random.normal(scale=1.0, size=3)
            pos += stepvec
            # clamp and deposit small energy
            ix = int(round(np.clip(pos[0],0,size-1)))
            iy = int(round(np.clip(pos[1],0,size-1)))
            iz = int(round(np.clip(pos[2],0,size-1)))
            grid[ix,iy,iz] += 1.0
            # stop with some probability (absorption)
            if np.random.rand() < 0.02:
                break
    # show central slice of photon density
    plt.figure(figsize=(4,4))
    plt.title("Phoxels: central slice of photon count grid")
    plt.imshow(grid[:,:,size//2], origin='lower')
    plt.axis('off')

# ------------------ 7) Toxel: temporal voxel (4D toy) ------------------
def demo_toxels():
    # create a time sequence of volumes where a gaussian blob moves with time
    size = 32
    T = 6
    vols = np.zeros((T,size,size,size))
    x = np.linspace(-1,1,size)
    X,Y,Z = np.meshgrid(x,x,x)
    for t in range(T):
        cx = 0.5*math.sin(2*math.pi*t/T)
        cy = 0.5*math.cos(2*math.pi*t/T)
        cz = 0.0
        vols[t] = np.exp(-(((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)/(2*0.18**2)))
    # visualize two timesteps to show temporal change
    plt.figure(figsize=(4,4))
    plt.title("Toxel: central slice at t=0")
    plt.imshow(vols[0,:,:,size//2], origin='lower')
    plt.axis('off')
    plt.figure(figsize=(4,4))
    plt.title("Toxel: central slice at t=T/2")
    plt.imshow(vols[T//2,:,:,size//2], origin='lower')
    plt.axis('off')

# ------------------ 8) Flexel: flexible sensor element grid ------------------
def demo_flexels():
    # create a regular sensor grid then warp it to simulate flexible substrate deformation
    H,W = 20,30
    Y,X = np.mgrid[0:H,0:W]
    # readings are a smooth function; warp coordinates
    warpX = X + 2.5*np.sin(2*np.pi*Y/10.0)
    warpY = Y + 1.5*np.cos(2*np.pi*X/15.0)
    # sample a signal at warped coordinates (toy scalar field)
    readings = np.sin(warpX/5.0) + 0.5*np.cos(warpY/4.0)
    plt.figure(figsize=(5,3))
    plt.title("Flexels: warped sensor grid readings")
    plt.imshow(readings, origin='lower', interpolation='nearest')
    plt.axis('off')

# ------------------ 9) VoxTexel: voxel storing texture (color per voxel) ------------------
def demo_voxtexel():
    # create voxel grid where each voxel stores an RGB texel (e.g., colored volume)
    size = 24
    xs = np.linspace(-1,1,size)
    X,Y,Z = np.meshgrid(xs,xs,xs)
    # color by position mapped to RGB-like triplet
    R = 0.5*(X+1.0)
    G = 0.5*(Y+1.0)
    B = 0.5*(Z+1.0)
    # threshold to get surface voxels and collect colors
    vol_density = np.exp(-((X**2+Y**2+Z**2)/(2*0.4**2)))
    mask = vol_density > 0.2
    pts = np.argwhere(mask)
    colors = np.zeros((pts.shape[0],3))
    for i,p in enumerate(pts):
        colors[i,0] = R[tuple(p)]
        colors[i,1] = G[tuple(p)]
        colors[i,2] = B[tuple(p)]
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("VoxTexel: colored voxels (scatter with stored texels)")
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=8)
    ax.set_box_aspect((1,1,1))

# --------- Run all demos ----------
demo_pixels()
demo_texels()
demo_voxels()
demo_surfels()
demo_spaxels()
demo_phoxels()
demo_toxels()
demo_flexels()
demo_voxtexel()

plt.show()

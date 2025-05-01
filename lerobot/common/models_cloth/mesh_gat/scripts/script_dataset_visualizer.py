import os, sys, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def plot_corresponding_meshes_and_images(batch, num_lines):
    """
    Plot each row with the true image and true mesh.

    Parameters:
    batch: Dictionary containing batch data.
    num_lines: Number of lines to plot.
    """
    fig = plt.figure(figsize=(15, 10))

    for i in range(num_lines):
        # Plot true image using OpenCV
        ax_img = fig.add_subplot(num_lines, 2, i * 2 + 1)
        image = batch['image_simu'][i]
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ax_img.imshow(image_bgr)
        ax_img.set_title(f'True Image {i + 1}')
        ax_img.axis('off')

        # Plot true mesh
        true_mesh = batch['mesh_simu'][i]
        ax_true_mesh = fig.add_subplot(num_lines, 2, i * 2 + 2, projection='3d')
        ax_true_mesh.scatter(true_mesh[:, 0], true_mesh[:, 1], true_mesh[:, 2], c='r', s=1)
        ax_true_mesh.set_title(f'True Mesh {i + 1}')
        ax_true_mesh.view_init(elev=90, azim=-90)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    name_folder = 'train'
    name_cloth = 't_shirt_l3'
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = f'/home/ktang/ws/data/mesh_gat/{name_cloth}/{name_folder}'
    
    img_path_list = sorted(glob.glob(os.path.join(DATASET_DIR, '*.render_depth.png')))
    data_size = len(img_path_list)
    
    # Define the number of lines to plot
    num_lines = 4
    num_loop = 30
    
    for i in range(num_loop):
        # Randomly select a batch of images and meshes
        batch_indices = random.sample(range(data_size), num_lines)
        batch = {'image_simu': [], 'mesh_simu': []}
        
        for idx in batch_indices:
            image_path = img_path_list[idx]
            mesh_path = image_path.replace('render_depth.png', 'simu_pcl.txt')
            try:
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mesh = np.loadtxt(mesh_path)
                batch['image_simu'].append(image_rgb)
                batch['mesh_simu'].append(mesh)
            except:
                print(f'Error loading data for index {idx}')
        
        # Convert lists to numpy arrays
        batch['image_simu'] = np.array(batch['image_simu'])
        batch['mesh_simu'] = np.array(batch['mesh_simu'])
        
        # Call the plotting function
        plot_corresponding_meshes_and_images(batch, num_lines)
import cv2
import matplotlib
import matplotlib.pyplot as plt

def show_process(current, total, prefix='Process', suffix='Complete', decimals=1, length=40, fill='█', printEnd="\r"):
    '''
    print iteration process bar in terminal
    '''
    # start process
    if current == 0:
        print('\n=================================== Process Started ===================================')
    # calculate process percent
    percent = ("{0:." + str(decimals) + "f}").format(100 * ((current + 1) / float(total)))
    # calculate process length
    filledLength = int(length * (current + 1) // total)
    # generate process bar
    bar = fill * filledLength + '-' * (length - filledLength)
    # print process bar
    print(f'{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # finish process
    if current + 1 == total:
        print('\n=================================== Process Finished ===================================')

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

def plot_image_mesh_pair(image, mesh):
    """
    Plot one row with the image and mesh.
    """
    fig = plt.figure()

    # Plot image using OpenCV
    ax_img = fig.add_subplot(0, 2, 1)
    loc_image = image
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ax_img.imshow(image_bgr)
    ax_img.set_title(f'Image')
    ax_img.axis('off')

    # Plot true mesh
    loc_mesh = mesh
    ax_true_mesh = fig.add_subplot(0, 2, 2, projection='3d')
    ax_true_mesh.scatter(loc_mesh[:, 0], loc_mesh[:, 1], loc_mesh[:, 2], c='r', s=1)
    ax_true_mesh.set_title(f'Mesh')
    ax_true_mesh.view_init(elev=90, azim=-90)

    plt.tight_layout()
    plt.show()

def plot_prediction_result_with_label(name, img_ori, img_trans, mesh_true, mesh_pred, save_path):
    """
    Plot one row with the images and meshs.
    """
    # Use non-interactive backend to avoid Tkinter issues
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure()

    # Plot image using OpenCV
    ax_img = fig.add_subplot(1, 4, 1)
    loc_image = img_ori
    image_bgr = cv2.cvtColor(loc_image, cv2.COLOR_RGB2BGR)
    ax_img.imshow(image_bgr)
    ax_img.set_title(f'Original Image')
    ax_img.axis('off')

    ax_img = fig.add_subplot(1, 4, 2)
    loc_image_trans = img_trans
    image_bgr_trans = cv2.cvtColor(loc_image_trans, cv2.COLOR_RGB2BGR)
    ax_img.imshow(image_bgr_trans)
    ax_img.set_title(f'Transform Image')
    ax_img.axis('off')

    # Plot true mesh
    loc_mesh = mesh_true
    ax_true_mesh = fig.add_subplot(1, 4, 3, projection='3d')
    ax_true_mesh.scatter(loc_mesh[:, 0], loc_mesh[:, 1], loc_mesh[:, 2], c='r', s=1)
    ax_true_mesh.set_title(f'True Mesh')
    ax_true_mesh.view_init(elev=90, azim=-90)
    
    # Plot predicted mesh
    loc_mesh_pred = mesh_pred
    ax_pred_mesh = fig.add_subplot(1, 4, 4, projection='3d')
    ax_pred_mesh.scatter(loc_mesh_pred[:, 0], loc_mesh_pred[:, 1], loc_mesh_pred[:, 2], c='r', s=1)
    ax_pred_mesh.set_title(f'Predicted Mesh')
    ax_pred_mesh.view_init(elev=90, azim=-90)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_result_without_true_label(name, img_ori, img_trans, mesh_pred, save_path):
    """
    Plot one row with the images and meshs.
    """
    # Use non-interactive backend to avoid Tkinter issues
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure()

    # Plot image using OpenCV
    ax_img = fig.add_subplot(1, 3, 1)
    loc_image = img_ori
    image_bgr = cv2.cvtColor(loc_image, cv2.COLOR_RGB2BGR)
    ax_img.imshow(image_bgr)
    ax_img.set_title(f'Original Image')
    ax_img.axis('off')

    ax_img = fig.add_subplot(1, 3, 2)
    loc_image_trans = img_trans
    image_bgr_trans = cv2.cvtColor(loc_image_trans, cv2.COLOR_RGB2BGR)
    ax_img.imshow(image_bgr_trans)
    ax_img.set_title(f'Transform Image')
    ax_img.axis('off')
    
    # Plot predicted mesh
    loc_mesh_pred = mesh_pred
    ax_pred_mesh = fig.add_subplot(1, 3, 3, projection='3d')
    ax_pred_mesh.scatter(loc_mesh_pred[:, 0], loc_mesh_pred[:, 1], loc_mesh_pred[:, 2], c='r', s=1)
    ax_pred_mesh.set_title(f'Predicted Mesh')
    ax_pred_mesh.view_init(elev=90, azim=-90)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_curve(loss_record, save_path):
    '''
    Function to plot and save the training/validation loss figure
    '''
    # Use non-interactive backend to avoid Tkinter issues
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(loss_record['train'], label='Train Loss', marker='o', linestyle='-')
    plt.plot(loss_record['val'], label='Validation Loss', marker='s', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
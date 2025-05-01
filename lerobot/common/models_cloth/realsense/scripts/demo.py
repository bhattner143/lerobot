import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def get_pcd(pipeline, pcd, device, color_w, color_h):
    # Dummy example function
    return np.random.rand(10000, 3) * 0.1

# Create the renderer once
render_width, render_height = 512, 512
renderer = o3d.visualization.rendering.OffscreenRenderer(
    width=render_width, 
    height=render_height
)

# Set some general scene properties that won't need changing each iteration
renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
renderer.scene.scene.enable_sun_light(True)
renderer.scene.scene.set_sun_light(
    direction=[0.0, 0.0, -1.0],
    intensity=75000,
    color=[1.0, 1.0, 1.0]
)

t_total = 0.0
num_frame = 0

for i in range(10):  # for example, rendering 10 frames
    t0 = time.time()

    # 1) Build or update your geometry
    pcd_np = get_pcd(None, None, None, None, None)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction([0, 0, 1])

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist/2, avg_dist, avg_dist*2, avg_dist*2.5]
    tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    tri_mesh.compute_triangle_normals()

    # 2) Remove any previously added geometry from the scene
    #    so we can re-add the new geometry for each loop iteration.
    if renderer.scene.has_geometry("mesh"):
        renderer.scene.remove_geometry("mesh")

    # 3) Add updated mesh with a material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", tri_mesh, material)

    # 4) Setup or update the camera each loop
    bbox = tri_mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center().astype(np.float32)
    eye = center + np.array([0.0, 0.0, 0.8], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov_degrees = 40.0
    renderer.setup_camera(
        vertical_field_of_view=fov_degrees,
        center=center,
        eye=eye,
        up=up,
        near_clip=0.01,
        far_clip=100.0
    )

    # 5) Render depth
    #    If you want real distances, set z_in_view_space=True instead
    depth_o3d = renderer.render_to_depth_image(z_in_view_space=False)
    depth_np = np.asarray(depth_o3d)

    t1 = time.time()
    t_total += t1 - t0
    num_frame += 1
    print(f"Time to get PCD + render depth: {t1 - t0:.2f}s")
    print(f"FPS: {1./(t1 - t0):.2f}")
    print(f"avg FPS: {num_frame / t_total:.2f}")

    plt.imshow(depth_np, cmap='gray')
    plt.title("Depth Image")
    plt.colorbar()
    plt.show()

# After the loop, optionally delete the renderer
del renderer
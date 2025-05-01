import os

def rename_files(directory, initial_index, new_start_index):
    
    # depth files
    # List all .simu_mesh.obj files in the directory
    all_files_dep = sorted([f for f in os.listdir(directory) if f.endswith('.render_depth.png')])
    # Filter files based on the initial index
    # selected_files_dep = [f for f in all_files_dep if int(f.split('.')[0]) >= initial_index]
    n = len(all_files_dep)
    # Ensure we have enough files to rename
    # if len(selected_files_dep) < n:
    #     print(f"Not enough files to rename. Found {len(selected_files_dep)} files starting from index {initial_index}.")
    #     return
    # Select the first n files
    selected_files_dep = all_files_dep[:n]
    # Rename the selected files with new consecutive indices starting from new_start_index
    for i, old_file in enumerate(selected_files_dep):
        old_file_path = os.path.join(directory, old_file)
        new_file_name = f"{new_start_index + i:06d}.render_depth.png"
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file} to {new_file_name}")
    
    # obj files
    # List all .simu_mesh.obj files in the directory
    all_files_obj = sorted([f for f in os.listdir(directory) if f.endswith('.simu_mesh.obj')])
    # Filter files based on the initial index
    selected_files_obj = [f for f in all_files_obj if int(f.split('.')[0]) >= initial_index]
    n = len(selected_files_obj)
    # Ensure we have enough files to rename
    if len(selected_files_obj) < n:
        print(f"Not enough files to rename. Found {len(selected_files_obj)} files starting from index {initial_index}.")
        return
    # Select the first n files
    selected_files_obj = selected_files_obj[:n]
    # Rename the selected files with new consecutive indices starting from new_start_index
    for i, old_file in enumerate(selected_files_obj):
        old_file_path = os.path.join(directory, old_file)
        new_file_name = f"{new_start_index + i:06d}.simu_mesh.obj"
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file} to {new_file_name}")


    # txt files
    # List all .simu_pcl.txt files in the directory
    all_files_txt = sorted([f for f in os.listdir(directory) if f.endswith('.simu_pcl.txt')])
    # Filter files based on the initial index
    selected_files_txt = [f for f in all_files_txt if int(f.split('.')[0]) >= initial_index]
    n = len(selected_files_txt)
    # Ensure we have enough files to rename
    if len(selected_files_txt) < n:
        print(f"Not enough files to rename. Found {len(selected_files_txt)} files starting from index {initial_index}.")
        return
    # Select the first n files
    selected_files_txt = selected_files_txt[:n]
    # Rename the selected files with new consecutive indices starting from new_start_index
    for i, old_file in enumerate(selected_files_txt):
        old_file_path = os.path.join(directory, old_file)
        new_file_name = f"{new_start_index + i:06d}.simu_pcl.txt"
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file} to {new_file_name}")

    # rgb files
    # List all .simu_rgb.png files in the directory
    all_files_png = sorted([f for f in os.listdir(directory) if f.endswith('.simu_rgb.png')])
    # Filter files based on the initial index
    selected_files_png = [f for f in all_files_png if int(f.split('.')[0]) >= initial_index]
    # Ensure we have enough files to rename
    if len(selected_files_png) < n:
        print(f"Not enough files to rename. Found {len(selected_files_png)} files starting from index {initial_index}.")
        return
    # Select the first n files
    selected_files_png = selected_files_png[:n]
    # Rename the selected files with new consecutive indices starting from new_start_index
    for i, old_file in enumerate(selected_files_png):
        old_file_path = os.path.join(directory, old_file)
        new_file_name = f"{new_start_index + i:06d}.simu_rgb.png"
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file} to {new_file_name}")


# Example usage
directory = '/home/ktang/ws/data/mesh_gat/t_shirt_l3/source/t_shirt_l3_candidate_3'  # Replace with your directory path
initial_index = 0
new_start_index = 84061

rename_files(directory, initial_index, new_start_index)
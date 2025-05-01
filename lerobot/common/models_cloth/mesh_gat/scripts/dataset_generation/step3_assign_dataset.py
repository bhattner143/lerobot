import os

def find_files_with_suffix(root_folder, suffix, output_file):
    matching_files = []

    # Walk through all directories and subdirectories
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(suffix):  # Check if filename ends with the given suffix
                full_path = os.path.join(dirpath, filename)
                matching_files.append(full_path)

    # Write results to a text file
    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in matching_files:
            f.write(file_path + "\n")

    print(f"Search complete! Found {len(matching_files)} files. Results saved in '{output_file}'.")

    return matching_files

if __name__ == "__main__":

    # get all the data files
    suffix = ".render_depth.png"
    folder_path = '/home/ktang/ws/data/mesh_gat/t_shirt_l3/source'
    save_address = os.path.join(folder_path, "output.txt")
    files_name = find_files_with_suffix(folder_path, suffix=suffix, output_file=save_address)

    # clean all the files om the train val and test folders
    train_folder = '/home/ktang/ws/data/mesh_gat/t_shirt_l3/train'
    val_folder = '/home/ktang/ws/data/mesh_gat/t_shirt_l3/val'
    test_folder = '/home/ktang/ws/data/mesh_gat/t_shirt_l3/test'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    if not os.path.exists(test_folder): 
        os.makedirs(test_folder)

    for folder in [train_folder, val_folder, test_folder]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleaned {folder}")

    # # select 60% of the files for training, 20% for validation, and 20% for testing
    # train_files = files_name[:int(len(files_name)*0.6)]
    # val_files = files_name[int(len(files_name)*0.6):int(len(files_name)*0.8)]
    # test_files = files_name[int(len(files_name)*0.8):]
    # num_train = len(train_files)
    # num_val = len(val_files)
    # num_test = len(test_files)
    # print(f"Train: {num_train}, Val: {num_val}, Test: {num_test}")

    # do it circularly
    train_files = []
    val_files = []
    test_files = []
    num_files = len(files_name)
    for i in range(num_files):
        if i % 10 == 0 or i % 10 == 1:
            val_files.append(files_name[i])
        elif i % 10 == 2:
            test_files.append(files_name[i])
        else:
            train_files.append(files_name[i])
    num_train = len(train_files)
    num_val = len(val_files)
    num_test = len(test_files)
    print(f"Train: {num_train}, Val: {num_val}, Test: {num_test}")

    num_copy_train = 0
    # copy the files to the train, val, and test folders
    for file in train_files:
        pcl_file = file[:-16] + 'simu_pcl.txt'
        depth_name = file.split("/")[-1]
        pcl_name = depth_name[:-16] + 'simu_pcl.txt'
        os.system(f"cp {file} {os.path.join(train_folder, depth_name)}")
        os.system(f"cp {pcl_file} {os.path.join(train_folder, pcl_name)}")
        num_copy_train += 1
    print(f"Copied {num_copy_train} files to train folder.")

    num_copy_val = 0
    for file in val_files:
        pcl_file = file[:-16] + 'simu_pcl.txt'
        depth_name = file.split("/")[-1]
        pcl_name = depth_name[:-16] + 'simu_pcl.txt'
        os.system(f"cp {file} {os.path.join(val_folder, depth_name)}")
        os.system(f"cp {pcl_file} {os.path.join(val_folder, pcl_name)}")
        num_copy_val += 1
    print(f"Copied {num_copy_val} files to val folder.")

    num_copy_test = 0
    for file in test_files:
        pcl_file = file[:-16] + 'simu_pcl.txt'
        depth_name = file.split("/")[-1]
        pcl_name = depth_name[:-16] + 'simu_pcl.txt'
        os.system(f"cp {file} {os.path.join(test_folder, depth_name)}")
        os.system(f"cp {pcl_file} {os.path.join(test_folder, pcl_name)}")
        num_copy_test += 1
    print(f"Copied {num_copy_test} files to test folders.")

    print(f"Total: {num_copy_train + num_copy_val + num_copy_test} files copied.")

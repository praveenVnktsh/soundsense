import os

# Specify the folder path
data_path = '/home/punygod_admin/SoundSense/soundsense/data/mulsa/data'

## the path has several folders inside it
# Get the list of folders in the data path
folders = sorted(os.listdir(data_path))

for folder in folders:
  folder_path = os.path.join(data_path, folder)
  folder_path = os.path.join(folder_path, 'video')

  # Get the list of files in the folder
  files = sorted(os.listdir(folder_path))

  # Iterate over each file
  i = 1
  for file_name in files:
    # Generate the new file name
    new_file_name =  "{:05d}.png".format(i)

    # Generate the full paths for the old and new files
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_file_name)

    # Rename and save the file
    os.rename(old_file_path, new_file_path)

    print(f'Renamed {file_name} to {new_file_name} and saved it.')
    i+=1

# # Get the list of files in the folder
# files = sorted(os.listdir(folder_path))

# # Iterate over each file
# i = 1
# for file_name in files:
#   # Generate the new file name
#   new_file_name =  "{:05d}.png".format(i)

#   # Generate the full paths for the old and new files
#   old_file_path = os.path.join(folder_path, file_name)
#   new_file_path = os.path.join(folder_path, new_file_name)

#   # Rename and save the file
#   os.rename(old_file_path, new_file_path)

#   print(f'Renamed {file_name} to {new_file_name} and saved it.')
#   i+=1
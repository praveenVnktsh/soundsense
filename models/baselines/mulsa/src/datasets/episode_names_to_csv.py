import os
import csv

folder_list = []
for folder in sorted(os.listdir('/home/punygod_admin/SoundSense/soundsense/data/mulsa/data')):
    folder_list.append(folder)
# print(sorted(folder_list))
folder_list = sorted(folder_list, key=lambda x: int(x))
# print(folder_list)
with open('episode_names_to_csv.csv', 'w', newline='\n') as file:
    writer = csv.writer(file)

    for folder in folder_list:
        writer.writerow([folder])

file.close()
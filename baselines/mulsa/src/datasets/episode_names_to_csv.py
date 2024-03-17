import os
import csv

folder_list = []
for folder in sorted(os.listdir('/home/punygod_admin/SoundSense/soundsense/data/playbyear_runs')):
    folder_list.append(folder)

with open('episode_names_to_csv.csv', 'w', newline='\n') as file:
    writer = csv.writer(file)

    for folder in folder_list:
        writer.writerow([folder])

file.close()
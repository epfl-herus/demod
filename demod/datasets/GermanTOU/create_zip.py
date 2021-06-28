"""Creates Zip files that are uploaded on github for parsed data download."""
import shutil
import os

real_path = os.path.dirname(os.path.realpath(__file__))

activity_type = input('Which activity type ? ')

parsed_act_dir = os.path.join(real_path, 'parsed_data', activity_type)
zip_file = os.path.join(real_path, 'parsed_data', activity_type)

if os.path.isdir(parsed_act_dir):

    shutil.make_archive(zip_file, 'zip', parsed_act_dir)
else:
    print('No such paths :', parsed_act_dir)
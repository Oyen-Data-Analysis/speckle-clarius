import os

for subdir, dirs, files in os.walk("Analysis"):
    print(subdir)
    # print(dirs)
    # print(files)
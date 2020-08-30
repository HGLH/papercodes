import glob
import os
root="./datasets/horse2zebra"
mode="trainA"
#files= sorted(glob.glob(os.path.join(root,"%sA" %mode) + "/*.*"))
#files = sorted(glob.glob(os.path.join(root,mode) + "/*.*"))
files=os.path.join(root,mode)
files1=os.path.join(root,"%sA" %mode)
print(files)
print(files1)
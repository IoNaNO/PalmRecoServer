import os

for root, dirs, files in os.walk('./Users'):
    print(root, dirs)

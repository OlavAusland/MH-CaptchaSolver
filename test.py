import os
import random
import shutil


def main():
    files = os.listdir('./annotated')
    random.shuffle(files)

    [shutil.copy(f'./annotated/{file}', f'./captcha/{file}') for file in files[0:25]]

main()
# Install dependencies (python modules)

# Compile Metro: Done
import argparse
from os import makedirs, system
from shutil import rmtree
from os.path import dirname, realpath, join, exists, isfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    cur_dir = dirname(realpath(__file__))
    build_dir = join(cur_dir, "auxiliary", "metro_sources", "build")
    if args.build:
        if not exists(build_dir):
            makedirs(build_dir)
        system(f"cd {build_dir};echo $PWD; cmake ..; make")
    elif args.clean:
        if exists(build_dir):
            rmtree(build_dir)



if __name__ == '__main__':
    main()

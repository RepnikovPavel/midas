from distutils.core import setup

setup(
    name='pandaset',
    version='0.3dev',
    install_requires=['pandas','opencv-python','transforms3d','tqdm'],
    packages=['pandaset'],
)

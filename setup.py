from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkdetection*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkdetection',
    version='1.0.5',
    description='my object detection library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukingh01/kkdetection",
    author='kazuking',
    author_email='kazukingh01@gmail.com',
    license='Public License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Private License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@c9cf7c91c454bb875705af1b7f01e2a15da0d18f',
        'kkannotation @ git+https://github.com/kazukingh01/kkannotation.git@da938d026a8ad6edf1e556efa010067dc488fc50',
        'kkimgaug @ git+https://github.com/kazukingh01/kkimgaug.git@d4a715b4ff25988ce4b17324f9843c04dc99fd1a',
        'paddledet @ git+https://github.com/PaddlePaddle/PaddleDetection.git@eaf2dbe091d79f4824329ea885503e5f08fbb7ec',
        'Cython>=0.29.24',
        'cython-bbox>=0.1.3',
        'lap>=0.4.0',
    ],
    python_requires='>=3.8'
)

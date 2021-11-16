from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkdetection*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkdetection',
    version='1.0.1',
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
        'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@d1e04565d3bec8719335b88be9e9b961bf3ec464',
        'kkannotation @ git+https://github.com/kazukingh01/kkannotation.git@35e5169dd8fd5999077ea2e6c78b1e51f1643586',
        'kkimgaug @ git+https://github.com/kazukingh01/kkimgaug.git@5c4924fc7bdae4fafa1d3e57ec878e657ca8bcd1',
        'paddledet @ git+https://github.com/PaddlePaddle/PaddleDetection.git@a769ae3a38509b4f143a9dc0a28e0ec9b153fb1d',
        'Cython>=0.29.24',
        'cython-bbox>=0.1.3',
        'lap>=0.4.0',
    ],
    python_requires='>=3.8'
)

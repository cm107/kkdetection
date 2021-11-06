from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkdetectron2*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkdetectron2',
    version='1.0.0',
    description='my detectron2 library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukingh01/kkdetectron2",
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
        'kkannotation @ git+https://github.com/kazukingh01/kkannotation.git',
        'kkimgaug @ git+https://github.com/kazukingh01/kkimgaug.git',
        'cython-bbox>=0.1.3',
        'lap>=0.4.0',
    ],
    python_requires='>=3.8'
)

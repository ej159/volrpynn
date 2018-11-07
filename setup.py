import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="volrpynn",
    version="0.0.1",
    author="Jens Egholm Pedersen",
    author_email="jensegholm@protonmail.com",
    description="Library for modelling and training spiking neural networks through PyNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/volr/volrpynn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPLv3",
        "Operating System :: OS Independent",
    ],
)

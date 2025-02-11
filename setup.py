import glob
from setuptools import setup, find_packages

setup(
    name="snowwi_tools",
    version="0.1.0",
    author="Marc Closa Tarres (MCT)",
    author_email="mclosatarres@umass.edu",
    description="A collection of tools for SNOWWI processing and data handling.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mclosa8/snowwi-tools",
    packages=find_packages(),
    scripts=glob.glob('bin/*.py'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True
)


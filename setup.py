import glob
import yaml

from setuptools import setup, find_packages

def parse_environment_yml(file_path="env/environment.yml"):
    """Read dependencies from environment.yml and return a list of pip-compatible packages."""
    with open(file_path, "r") as file:
        env_data = yaml.safe_load(file)
    
    dependencies = []
    for dep in env_data.get("dependencies", []):
        # Handle pip section separately if present
        if isinstance(dep, dict) and "pip" in dep:
            dependencies.extend(dep["pip"])
        elif isinstance(dep, str):
            # Exclude python version dependency for pip packages
            if not dep.startswith("python"):
                dependencies.append(dep)
    return dependencies

setup(
    name="snowwi-tools",
    version="0.1.0",
    author="Marc Closa Tarres (MCT)",
    author_email="your.email@example.com",
    description="A collection of tools for SNOWWI processing and data handling.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mclosa8/snowwi-tools",  # Replace with the actual repo URL if available
    packages=find_packages(where="snowwi_tools"),
    package_dir={"": "."},
    scripts=glob.glob('bin/*.py'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=parse_environment_yml(),
    include_package_data=True,
)

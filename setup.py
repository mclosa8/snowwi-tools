import glob
from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='snowwi_lite',
        version='0.1',
        author='Marc Closa',
        packages=find_packages(),
        scripts=glob.glob('bin/*.py') + glob.glob('bin/*.sh') + glob.glob('tools/*.py'),
    )
from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    author='Michael Pakhmurin',
    url='https://github.com/MehMessGo/simple-ar',
    license='MIT License, see LICENSE file',
    author_email='mehmessgo@gmail.com',
    name='simple_ar',
    version='1.0',
    packages=find_packages(),
    long_description=open('README.md').read(),
    install_requires=['numpy', 'opencv-python']
)

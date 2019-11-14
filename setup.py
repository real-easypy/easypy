#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='weka-easypy',
    version='0.4.0',
    description='easypy is a collection of python modules that makes developers happy',
    author='Ofer Koren',
    author_email='koreno@gmail.com',
    url='https://github.com/weka-io/easypy',
    license='BSD',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'eziplog=easypy.ziplog:main',
            'ezcolorize=easypy.colors:main',
        ]
    },
)


# how to upload a package:
# 0. increment the version above
# 1. python3 setup.py sdist bdist_wheel
# 2. python3 -m twine upload dist/*

"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup

APP = ['../YViewer.py']
DATA_FILES = []
OPTIONS = {
    'resources': ['../icon.tiff'],
    'iconfile': '../icon.tiff'
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
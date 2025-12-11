"""
py2app build script for WTF Am I Doing?
Usage: python setup.py py2app
"""

from setuptools import setup

APP = ['src/wtf_am_i_doing/main.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'iconfile': None,
    'plist': {
        'CFBundleName': 'WTF Am I Doing',
        'CFBundleDisplayName': 'WTF Am I Doing?',
        'CFBundleIdentifier': 'com.wtfamidoing.app',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'NSHighResolutionCapable': True,
        'NSScreenCaptureUsageDescription': 'WTF Am I Doing needs screen recording permission to capture screenshots and describe your activity.',
        'LSMinimumSystemVersion': '10.15',
    },
    'packages': ['wtf_am_i_doing', 'mss', 'PIL'],
    'includes': ['tkinter'],
}

setup(
    name='WTF Am I Doing',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

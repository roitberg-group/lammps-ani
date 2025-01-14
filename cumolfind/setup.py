from setuptools import setup, find_packages

setup(
    name='cumolfind',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'PubChemPy',
    ],
    entry_points={
        'console_scripts': [
            'cumolfind-molfind=cumolfind.molfind:main',
            'cumolfind-split_traj=cumolfind.split_traj:main',
        ],
    },
)


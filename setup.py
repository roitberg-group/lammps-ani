from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lammps_ani',
    description='Lammps ANI interface',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/roitberg-group/lammps-ani',
    author='Roitberg Group',
    license='Apache License 2.0',
    author_email='jinzexue@ufl.edu',
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm']
)

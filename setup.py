from setuptools import setup, find_packages

# Layout & format inspired from https://github.com/navdeep-G/samplemod/blob/master/setup.py

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Feyn-plots',
    version='0.1.0',
    description='Charting Feyn graphs',
    long_description=readme,
    author='Chris Cave',
    author_email='chris.cave@abzu.ai',
    url='https://github.com/chriscave/cave-plots',
    license=license,
    packages=find_packages()

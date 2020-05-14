from setuptools import setup, find_packages

# Layout & format inspired from https://github.com/navdeep-G/samplemod/blob/master/setup.py



with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

requirements = ['feyn == 1.1.*']

setup(
    name='feynplots',
    version='0.0.2',
    description='Charting Feyn graphs',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Chris Cave',
    author_email='chris.cave@abzu.ai',
    url='https://github.com/chriscave/cave-plots',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements
)

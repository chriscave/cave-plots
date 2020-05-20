from setuptools import setup, find_packages

# Layout & format inspired from https://github.com/navdeep-G/samplemod/blob/master/setup.py



with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

requirements = ['feyn == 1.1.*', 'numpy >= 1.18.1', 'matplotlib >= 3.1.3']

setup(
    name='feynplots',
    version='1.0.3',
    description='Plotting Feyn graphs',
    long_description='Feynplots is a package to visualise each interaction in a Feyn Graph. Find the [documentation here](https://github.com/chriscave/cave-plots). See [here for more on starting with Feyn](https://docs.abzu.ai/docs/guides/quick_start.html) and [understanding the QLattice.](https://docs.abzu.ai/docs/guides/qlattice.html)',
    long_description_content_type='text/markdown',
    author='Chris Cave',
    author_email='chris.cave@abzu.ai',
    url='https://github.com/chriscave/cave-plots',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements
)

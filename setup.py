import setuptools

setuptools.setup(
    name='OctLearn',
    version='1',
    packages=['OctLearn', 'OctLearn.connector'],
    url='',
    license='',
    author='Kaidong Hu',
    author_email='',
    description='',
    install_requires=[
        'matplotlib>=3.3',
        'numpy>=1.19',
        'pymongo>=3.11',
        'torch>=1.15'
    ],
    scripts=['scripts/Train2.py']
)

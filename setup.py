from setuptools import find_packages, setup


setup(
    name='usvlib4ros',
    version='1.1.1.1',
    packages=find_packages(
        include=["usvlib4ros", "usvlib4ros.*"],
    ),
    url='',
    license='',
    install_requires=[
        "roslibpy==1.6.0",
        "numpy==2.2.6",
        "torch==2.12.1",
    ],
    author='qianlon',
    author_email='',
    description='',

)

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='boston_housing',
    version='0.1',
    author = 'Jaydev Trivedi',
    author_email = 'contactme@jaydevtrivedi.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Basic Boston Housing',
    requires=[]
)


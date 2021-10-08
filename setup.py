from setuptools import setup

setup(
    name='rtk',
    version='0.0.3',
    author='Ryo Ito, Calder Sheagren',
    packages=['rtk',],
    # license='LICENSE',
    description='LDDMM Registration Package',
    long_description=open('README.md').read(),
    include_package_data=True,
    install_requires=['numpy', 'nibabel', 'scikit-image', 'pydicom'],
)

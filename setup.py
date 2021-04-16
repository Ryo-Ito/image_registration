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
    data_files = [('man/man1', ['docs/build/man/rtk.1'])],
    install_requires=['numpy', 'nibabel', 'scikit-image', 'pydicom'],
)

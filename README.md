# LDDMM PYTHON

Original repo by @Ryo-ito, reorganized and ported to `python3` by @calderds

To install:

First, clone the repository:

```bash
git clone https://gitlab.com/tdinav-pipeline/registration-tools
```

Now, we need to compile the `imageprocessing.pyx` file for your operating system and rename it so the package can see it. To do so (on a UNIX system), use the following commands.

```bash
cd rtk
python setup.py build_ext -i
mv imageprocessing*.so imageprocessing.so
```

Then, we can install the package like usual.

```bash
cd ..
python setup.py sdist
pip install dist/rtk-0.0.1.tar.gz
```

Now, to test the repository, we can use the test script provided to make sure the installation works on your system.

```bash
cd example
python cube_to_sphere.py
```

Or, you can use tools in the `cli` folder to directly call this toolbox from the command line.

## Original README: 


# image_registration
Image registration package for python.
Performs non-rigid deformation to match input image to another one.
Algorithms implemented are LDDMM(Large Deformation Diffeomorphic Metric Mapping) and SyN (Symmetric Normalization).
Supports for both 2 and 3 dimensional images in Nifti files.

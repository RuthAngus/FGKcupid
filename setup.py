from setuptools import setup

setup(name='fgkcupid',
      version='0.1',
      description='Online dating for stars.',
      url='http://github.com/RuthAngus/fgkcupid',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['fgkcupid'],
      include_package_data=True
      install_requires=["numpy", "matplotlib", "pandas", "os", "george",
                        "h5py"],
      zip_safe=False)

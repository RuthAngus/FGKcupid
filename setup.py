from setuptools import setup

setup(name='fgkcupid',
      version='0.1',
      description='Online dating for stars.',
      url='http://github.com/RuthAngus/fgkcupid',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['fgkcupid'],
      install_requires=['numpy', 'matplotlib', 'pandas', 'os', 'george',
                        'h5py'],
      include_package_data=True,
      zip_safe=False)

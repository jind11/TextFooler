from setuptools import setup


setup(name='ESIM',
      version=1.0,
      url='https://github.com/coetaur0/ESIM',
      license='Apache 2',
      author='Aurelien Coet',
      author_email='aurelien.coet19@gmail.com',
      description='Implementation in Pytorch of the ESIM model for NLI',
      packages=[
        'esim'
      ],
      install_requires=[
        'numpy',
        'nltk',
        'matplotlib',
        'tqdm',
        'torch'
      ])

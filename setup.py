from setuptools import setup, find_packages

setup(name='looptune',
      version='0.1',
      description='',
      url='https://github.com/japarty/looptune',
      author='Jakub Partyka',
      author_email='jakubpart@gmail.com',
      # license='MIT',
      packages=find_packages(),
      install_requires=[line.strip() for line in open('requirements.txt')],
      zip_safe=False)
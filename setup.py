from setuptools import setup

setup(name='gym_battleship',
      version='0.0.1',
      install_requires=[
            'docopt',
            'gym[classic_control]',
            'numpy',
            'tensorflow',
      ]
)
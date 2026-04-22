from setuptools import setup
import os
from glob import glob

package_name = 'basketball_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'models', 'basketball'), glob('models/basketball/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xavier',
    maintainer_email='xavier@example.com',
    description='Basketball defender simulation project',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'defender_controller = basketball_project.defender_controller:main',
            'spawn_random_ball = basketball_project.spawn_random_ball:main',
            'ball_respawner = basketball_project.ball_respawner:main',
            'scorer_controller = basketball_project.scorer_controller:main',
	],
    },
)

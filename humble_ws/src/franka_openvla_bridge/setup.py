from setuptools import setup
import os
from glob import glob

package_name = 'franka_openvla_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Bridge package for OpenVLA and Franka robot control',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'franka_openvla_bridge = franka_openvla_bridge.franka_openvla_bridge:main',
        ],
    },
    python_requires='>=3.8',
) 
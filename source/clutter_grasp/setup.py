"""Installation script for the 'clutter_grasp' package (pip install -e source/clutter_grasp)."""

from setuptools import find_packages, setup

setup(
    name="clutter_grasp",
    version="0.1.0",
    description="A protocol for evaluating robotic grasping in clutter (Isaac Lab environments, clutter classifier).",
    packages=find_packages(include=["clutter_grasp", "clutter_grasp.*"]),
    package_data={"clutter_grasp": ["assets/**/*"]},
    include_package_data=True,
    install_requires=["stable-baselines3", "opencv-python", "pillow", "scipy", "segment-anything"],
    python_requires=">=3.10",
    license="BSD-3-Clause",
    zip_safe=False,
)

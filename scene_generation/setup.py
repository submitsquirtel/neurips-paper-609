from setuptools import setup, find_packages

setup(
    name='scene_generation',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Add your dependencies here
        'numpy',
        'scikit-learn',
        'matplotlib',
        'opencv-python',
        'pyyaml',
        
    ],
)
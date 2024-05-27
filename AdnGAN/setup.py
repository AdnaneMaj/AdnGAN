from setuptools import setup, find_packages

setup(
    name='AdnGAN',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
    ],
    author='Adnane MAJDOUB',
    author_email = 'adnanemajdoub@gmail.com',
    description='AdnGAN is a Generative Adversarial Network (GAN) that generates images from random noise.',
    long_description=open("README.md").read(),
    url = "https://gitub.com/AdnaneMajdoub/AdnGAN",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

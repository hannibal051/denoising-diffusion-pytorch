from setuptools import setup, find_packages

setup(
  name = 'env_gen_diffusion',
  packages = find_packages(),
  version='1.0',
  license='MIT',
  description = 'Environment Generation Denoising Diffusion Probabilistic Models - Pytorch',
  author = 'Youwei Yu',
  author_email = 'yyw13532@gmail.com',
  url = 'https://google.com',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'generative models'
  ],
  install_requires=[
    'numpy'
  ],
)

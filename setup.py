from setuptools import setup, find_packages

setup(
    name='my-ml-package',
    version='0.1.0',
    author='Your Name',
    description='My Machine Learning Package',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'transformers',
        'torch',
        'langchain',
        'openai',
        'faiss-cpu',
        'huggingface_hub',
        'langchain-community',
        'langchain-openai',
        'llama-index-embeddings-huggingface',
        'llama-index-embeddings-instructor',
        'langchain-text-splitters'
    ],
)
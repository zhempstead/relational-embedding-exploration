from setuptools import setup, find_namespace_packages

setup(
    name='relational_embeddings',
    version='0.0.1',
    packages=find_namespace_packages(include=["relational_embeddings", "hydra_plugins.*"]),
)

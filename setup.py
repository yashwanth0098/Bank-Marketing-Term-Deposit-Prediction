from setuptools import setup, find_packages
from typing import List

def get_requirements()->List[str]:
    requirements_lst:List[str]=[]
    try:
        with open('requirements.txt') as file:
            lines = file.readlines()
            for line in lines:
                requirements=line.strip()
                if requirements and requirements!='-e .':
                    requirements_lst.append(requirements)
    except FileNotFoundError:
        print('requirements.txt file not found.')  
    return requirements_lst    

setup(
    name='mlproject',
    version='0.0.1',
    author='Yashwanth Kumar',
    author_email='YhM9S@example.com',
    packages=find_packages(),
    install_requires=get_requirements(),
)       
    
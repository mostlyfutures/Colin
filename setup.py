#!/usr/bin/env python3
"""
Setup script for Colin Trading Bot
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements_v2.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="colin-trading-bot",
    version="2.0.0",
    author="Colin Trading Bot Team",
    author_email="contact@colinbot.com",
    description="AI-powered institutional cryptocurrency trading system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/colin-trading-bot",
    packages=find_packages(exclude=["tests*", "tools*", "docs*", "archives*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
  extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "cli": [
            "click>=8.1.0",
            "rich>=13.0.0",
            "keyring>=24.0.0",
            "cryptography>=41.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "colin=colin_bot.cli.main:main",
            "colin-bot=colin_bot.v2.main:main",
            "colin-bot-v1=colin_bot.v1.main:main",
            "colin-bot-api=colin_bot.v2.api_gateway.rest_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "colin_bot": [
            "config/*.yaml",
            "config/*.json",
            "cli/*.yaml",
            "cli/*.json",
        ],
    },
    zip_safe=False,
)
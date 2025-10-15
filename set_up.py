# setup.py
from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).resolve().parent
req_file = root / "requirements.txt"


def read_requirements():
    if req_file.exists():
        lines = []
        for L in req_file.read_text(encoding="utf-8").splitlines():
            L = L.strip()
            if not L or L.startswith("#"):
                continue
            lines.append(L)
        return lines
    return []


setup(
    name="flow-grpo",
    version="0.0.1",
    description="Oscar (OSCAR: Orthogonal Stochastic Control for Alignment-Respecting Diversity in Flow Matching)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=read_requirements(),
    include_package_data=True,
)

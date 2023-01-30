import sys

sys.path.append("../test")


def pytest_addoption(parser):
    parser.addoption("--rank", action="store")

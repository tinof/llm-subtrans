import sys
import importlib.util


def check_required_imports(modules: list[str], pip_extras: str | None = None) -> None:
    """Check if required modules are available and exit with helpful message if not"""
    missing_modules = []
    for module_name in modules:
        if importlib.util.find_spec(module_name) is None:
            missing_modules.append(module_name)

    if missing_modules:
        print("Error: Required modules not found")
        print(f"Missing: {', '.join(missing_modules)}")
        print()
        if pip_extras:
            print("Installation options:")
            print(f"  - pipx install 'llm-subtrans[{pip_extras}]'")
            print(f"  - pip install '.[{pip_extras}]'")
            print(f"  - Run ./install.sh and select the {pip_extras} provider")
        else:
            print("Installation options:")
            print("  - pipx install llm-subtrans")
            print("  - pip install .")
            print("  - Run ./install.sh")
        sys.exit(1)

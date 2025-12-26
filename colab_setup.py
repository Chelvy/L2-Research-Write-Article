"""
Google Colab Setup Script for Integration Paradox Demonstration

This script handles dependency installation with proper conflict resolution
for Google Colab's pre-installed packages.

Usage:
    In a Colab cell, run:
    !wget -q https://raw.githubusercontent.com/.../colab_setup.py
    %run colab_setup.py

Or simply copy-paste this entire script into a Colab cell.
"""

import subprocess
import sys
import os

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def install_package(package_spec, upgrade=False):
    """Install a package with proper error handling."""
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package_spec)

    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to install {package_spec}")
        return False

def main():
    print_section("Integration Paradox Demo - Google Colab Setup")

    print("📦 Installing Core Dependencies...")
    print("This may take 2-3 minutes. Please be patient.\n")

    # Step 1: Upgrade pip
    print("1️⃣  Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    print("   ✅ pip upgraded\n")

    # Step 2: Install CrewAI and core dependencies
    print("2️⃣  Installing CrewAI framework...")
    core_packages = [
        "crewai>=0.80.0",
        "crewai-tools>=0.12.0",
    ]

    for pkg in core_packages:
        install_package(pkg)
    print("   ✅ CrewAI installed\n")

    # Step 3: Install LLM providers
    print("3️⃣  Installing LLM provider SDKs...")
    llm_packages = [
        "anthropic>=0.39.0",
        "openai>=1.54.0",
        "huggingface-hub>=0.26.0",
    ]

    for pkg in llm_packages:
        install_package(pkg)
    print("   ✅ LLM SDKs installed\n")

    # Step 4: Install LangChain integrations (with conflict handling)
    print("4️⃣  Installing LangChain integrations...")
    print("   (This may show warnings - they're safe to ignore)")

    langchain_packages = [
        "langchain-anthropic>=0.3.0",
        "langchain-openai>=0.2.0",
    ]

    for pkg in langchain_packages:
        install_package(pkg)
    print("   ✅ LangChain integrations installed\n")

    # Step 5: Install visualization libraries
    print("5️⃣  Installing visualization libraries...")
    viz_packages = [
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "plotly>=5.24.0",
    ]

    for pkg in viz_packages:
        install_package(pkg)
    print("   ✅ Visualization libraries ready\n")

    # Step 6: Verify critical imports
    print("6️⃣  Verifying installation...")

    verification_tests = [
        ("crewai", "CrewAI"),
        ("anthropic", "Anthropic"),
        ("openai", "OpenAI"),
        ("langchain_anthropic", "LangChain-Anthropic"),
        ("langchain_openai", "LangChain-OpenAI"),
        ("matplotlib", "Matplotlib"),
    ]

    all_passed = True
    for module, name in verification_tests:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            all_passed = False

    print()

    if all_passed:
        print_section("✅ SETUP COMPLETE!")
        print("All dependencies installed successfully.")
        print("You can now run the Integration Paradox demonstration.\n")
        print("Next steps:")
        print("1. Add your API keys to Colab Secrets (🔑 icon)")
        print("2. Run the notebook cells in order")
        print("3. Watch the Integration Paradox unfold!")
    else:
        print_section("⚠️  SETUP INCOMPLETE")
        print("Some packages failed to install.")
        print("Try running this script again, or install packages manually.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

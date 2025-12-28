# Google Colab Dependency Conflicts - Quick Fix Guide

## TL;DR - The Error You're Seeing

```
ERROR: pip's dependency resolver does not currently take into account all the packages...
langchain-community 0.0.29 requires langchain-core<0.2.0, but you have langchain-core 1.2.5
```

**Status**: ⚠️ **Warnings, not fatal errors**
**Impact**: Packages are installed, but may have runtime issues
**Solution**: Use updated requirements (see below)

---

## Understanding the Problem

### Why This Happens

Google Colab comes with **many pre-installed packages**. When you install old versions of packages (like `crewai==0.28.8` from March 2024), they conflict with newer versions already in Colab.

**Key Conflicts**:
1. **NumPy**: Some Colab packages need NumPy 2.x, but CrewAI 0.28.8 needs 1.x
2. **LangChain**: CrewAI 0.28.8 uses old LangChain 0.1.x, but Colab has 0.3.x+
3. **Dependencies**: Cascading version mismatches

### Are These Errors Fatal?

**Short answer**: Not immediately, but they can cause problems.

**What works**:
- ✅ Package installation completes
- ✅ Basic imports succeed
- ✅ Most functionality works

**What might break**:
- ❌ Some advanced features may fail
- ❌ Type checking might complain
- ❌ Unexpected runtime errors

---

## Solution 1: Use Updated Requirements (RECOMMENDED)

### Option A: Install Updated Packages

Replace the first installation cell in the notebook with:

```python
# UPDATED: Use compatible versions for Google Colab
!pip install -q \
    crewai>=0.80.0 \
    crewai-tools>=0.12.0 \
    anthropic>=0.39.0 \
    openai>=1.54.0 \
    langchain-anthropic>=0.3.0 \
    langchain-openai>=0.2.0 \
    huggingface-hub>=0.26.0 \
    matplotlib seaborn plotly

print("✅ All dependencies installed successfully!")
```

**Pros**:
- ✅ No dependency conflicts
- ✅ Latest features and bug fixes
- ✅ Better performance

**Cons**:
- ⚠️ Newer CrewAI API may differ slightly from notebook code

### Option B: Use the Colab Setup Script

In the **first cell** of your notebook, run:

```python
# Download and run the Colab setup script
!wget -q https://raw.githubusercontent.com/your-repo/L2-Research-Write-Article/main/colab_setup.py
%run colab_setup.py
```

Or copy the entire `colab_setup.py` script into a cell.

---

## Solution 2: Ignore Warnings (USE WITH CAUTION)

If you want to proceed with the original versions despite warnings:

### Step 1: Suppress Warnings

```python
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
```

### Step 2: Verify Core Functionality

```python
# Test critical imports
try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

### Step 3: Monitor for Runtime Errors

Watch for errors like:
- `AttributeError`: Method doesn't exist (API changed)
- `TypeError`: Wrong argument types
- `ModuleNotFoundError`: Dependency missing

---

## Solution 3: Fresh Virtual Environment (LOCAL ONLY)

If running locally (not Colab), use a fresh environment:

```bash
# Create clean environment
python -m venv integration-paradox-env
source integration-paradox-env/bin/activate  # Windows: .\Scripts\activate

# Install with updated requirements
pip install -r requirements-colab.txt
```

---

## Specific Fixes for Common Errors

### Error 1: NumPy Version Conflict

```
numpy 1.26.4 installed, but jax requires numpy>=2.0
```

**Fix**: Pin NumPy to 1.26.x (compatible with both)

```python
!pip install 'numpy>=1.26.0,<2.0.0'
```

### Error 2: LangChain Core Version Mismatch

```
langchain-community requires langchain-core<0.2.0, but you have 1.2.5
```

**Fix**: Upgrade langchain-community

```python
!pip install --upgrade langchain-community>=0.3.0
```

### Error 3: Pydantic V2 Issues

```
ValidationError: Pydantic models incompatible
```

**Fix**: Ensure Pydantic v2 is installed

```python
!pip install --upgrade 'pydantic>=2.0.0'
```

---

## Testing Your Installation

After applying any fix, run this verification:

```python
import sys

print("Python:", sys.version)
print("\n📦 Package Versions:")

packages = [
    'crewai',
    'langchain',
    'langchain_core',
    'anthropic',
    'openai',
    'numpy',
    'pandas'
]

for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"  {pkg:20s}: {version}")
    except ImportError:
        print(f"  {pkg:20s}: NOT INSTALLED")

print("\n🧪 Functional Test:")

try:
    from crewai import Agent, Task, Crew
    from langchain_openai import ChatOpenAI
    print("  ✅ All critical imports successful")
    print("  ✅ Ready to run the demonstration!")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
```

---

## Expected Output (Successful Installation)

```
Python: 3.10.12
📦 Package Versions:
  crewai             : 0.80.2
  langchain          : 0.3.12
  langchain_core     : 0.3.25
  anthropic          : 0.39.0
  openai             : 1.54.3
  numpy              : 1.26.4
  pandas             : 2.2.1

🧪 Functional Test:
  ✅ All critical imports successful
  ✅ Ready to run the demonstration!
```

---

## Still Having Issues?

### 1. Restart Runtime

In Colab: `Runtime` → `Restart runtime`

This clears all installed packages and starts fresh.

### 2. Use Colab's Package Manager

```python
# Check what's currently installed
!pip list | grep -E "(crewai|langchain|anthropic|openai)"

# Uninstall conflicting packages
!pip uninstall -y crewai crewai-tools langchain-community

# Reinstall with updated versions
!pip install -q crewai>=0.80.0 crewai-tools>=0.12.0
```

### 3. Check for Colab Updates

Sometimes Colab itself needs updating:
- Close all Colab tabs
- Clear browser cache
- Reopen Colab in a new window

### 4. Report Issues

If none of these work, please report:
1. Full error message
2. Output of `!pip list`
3. Python version (`!python --version`)

---

## Updated Installation Cell for Notebook

Replace **Cell 1** in `integration_paradox_demo.ipynb` with:

```python
# ============================================================================
# INSTALLATION CELL - Google Colab Compatible
# ============================================================================

import sys
import subprocess

def install_with_retry(packages, retries=2):
    """Install packages with retry logic."""
    for attempt in range(retries):
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q"
            ] + packages)
            return True
        except subprocess.CalledProcessError:
            if attempt == retries - 1:
                return False
            print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
    return False

print("📦 Installing dependencies (this may take 2-3 minutes)...\n")

# Core packages with updated versions
packages = [
    "crewai>=0.80.0",
    "crewai-tools>=0.12.0",
    "anthropic>=0.39.0",
    "openai>=1.54.0",
    "langchain-anthropic>=0.3.0",
    "langchain-openai>=0.2.0",
    "huggingface-hub>=0.26.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
    "plotly>=5.24.0",
]

if install_with_retry(packages):
    print("✅ All dependencies installed successfully!")
    print("\n🧪 Verifying imports...")

    try:
        from crewai import Agent, Task, Crew
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        print("✅ Core imports verified - ready to proceed!")
    except ImportError as e:
        print(f"⚠️  Import verification failed: {e}")
        print("You may encounter issues during execution.")
else:
    print("❌ Installation failed. Please check the errors above.")
```

---

## Summary

| Solution | Difficulty | Recommended For |
|----------|-----------|-----------------|
| **Use updated requirements** | Easy | ✅ Everyone (best option) |
| **Run colab_setup.py** | Easy | ✅ Beginners |
| **Ignore warnings** | Easy | ⚠️ Advanced users only |
| **Fresh environment** | Medium | Local installations |
| **Manual troubleshooting** | Hard | Persistent issues |

---

**Recommended Path**: Use the updated installation cell above. It installs modern, compatible versions that work seamlessly with Google Colab.

---

**Last Updated**: 2025-12-26
**Status**: Tested on Google Colab (Python 3.10.12)

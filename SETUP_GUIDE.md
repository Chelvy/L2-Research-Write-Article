# Setup Guide: Integration Paradox Demonstration

This guide provides detailed instructions for setting up the Integration Paradox demonstration in both Google Colab and local environments.

## Table of Contents

1. [Google Colab Setup (Recommended)](#google-colab-setup)
2. [Local Environment Setup](#local-environment-setup)
3. [API Key Configuration](#api-key-configuration)
4. [Troubleshooting](#troubleshooting)
5. [Cost Estimation](#cost-estimation)

---

## Google Colab Setup (Recommended)

### Why Google Colab?

- ✅ No local installation required
- ✅ Free GPU/TPU access
- ✅ Pre-installed Python packages
- ✅ Easy API key management via Secrets
- ✅ Shareable notebooks

### Step-by-Step Instructions

#### 1. Open Google Colab

Go to [https://colab.research.google.com/](https://colab.research.google.com/)

#### 2. Upload the Notebook

**Option A: Upload from GitHub**
```
File → Open notebook → GitHub tab →
Enter repository URL → Select integration_paradox_demo.ipynb
```

**Option B: Upload from Computer**
```
File → Upload notebook → Choose integration_paradox_demo.ipynb
```

#### 3. Configure API Keys

**IMPORTANT**: Never hardcode API keys in notebooks!

1. **Open Secrets Panel**:
   - Look for the 🔑 key icon on the left sidebar
   - Click to open "Secrets" panel

2. **Add OpenAI API Key**:
   - Click **"+ New secret"**
   - Name: `OPENAI_API_KEY` (exact, case-sensitive)
   - Value: Your OpenAI API key (starts with `sk-proj-...` or `sk-...`)
   - Toggle **"Notebook access"** to ON
   - Click **"Add"**

3. **Add Anthropic API Key**:
   - Click **"+ New secret"** again
   - Name: `ANTHROPIC_API_KEY` (exact, case-sensitive)
   - Value: Your Anthropic API key (starts with `sk-ant-...`)
   - Toggle **"Notebook access"** to ON
   - Click **"Add"**

4. **Add HuggingFace API Key**:
   - Click **"+ New secret"** again
   - Name: `HUGGINGFACE_API_KEY` (exact, case-sensitive)
   - Value: Your HuggingFace API key (starts with `hf_...`)
   - Toggle **"Notebook access"** to ON
   - Click **"Add"**

#### 4. Verify Configuration

Run this test cell to verify keys are accessible:

```python
from google.colab import userdata
import os

try:
    os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
    os.environ["ANTHROPIC_API_KEY"] = userdata.get('ANTHROPIC_API_KEY')
    os.environ["HUGGINGFACE_API_KEY"] = userdata.get('HUGGINGFACE_API_KEY')
    print("✅ All API keys configured successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check that all secrets are added and notebook access is enabled.")
```

#### 5. Run the Notebook

1. Click **"Runtime"** → **"Run all"**
2. The notebook will:
   - Install dependencies (~2-3 minutes)
   - Configure LLM models
   - Execute the SDLC pipeline
   - Generate metrics and visualizations

#### 6. Monitor Execution

- Progress appears in each cell's output
- Expected total runtime: 10-20 minutes
- Watch for error messages in red

---

## Local Environment Setup

### Prerequisites

- **Python**: 3.10 or higher
- **pip**: Latest version
- **Git**: For cloning the repository
- **Jupyter**: For running notebooks

### Installation Steps

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd L2-Research-Write-Article
```

#### 2. Create Virtual Environment (Recommended)

**Using venv**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda**:
```bash
conda create -n integration-paradox python=3.10
conda activate integration-paradox
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Expected installation time: 3-5 minutes

#### 4. Configure Environment Variables

**Option A: Using .env file (Recommended)**

Create a `.env` file in the project root:

```bash
# .env file (DO NOT COMMIT TO GIT)

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-your-actual-key-here

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here

# HuggingFace Configuration
HUGGINGFACE_API_KEY=hf_your-actual-key-here
```

Add `.env` to `.gitignore`:
```bash
echo ".env" >> .gitignore
```

**Option B: Export in Shell**

```bash
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export HUGGINGFACE_API_KEY="hf_..."
```

On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="sk-proj-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:HUGGINGFACE_API_KEY="hf_..."
```

#### 5. Launch Jupyter

```bash
jupyter notebook integration_paradox_demo.ipynb
```

Browser will open automatically at `http://localhost:8888`

#### 6. Update API Configuration Cell

If using `.env` file, modify the API configuration cell:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify keys are loaded
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found"
assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not found"
assert os.getenv("HUGGINGFACE_API_KEY"), "HUGGINGFACE_API_KEY not found"

print("✅ All API keys loaded successfully!")
```

---

## API Key Configuration

### How to Obtain API Keys

#### OpenAI API Key

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in or create account
3. Click **"Create new secret key"**
4. Name it (e.g., "Integration Paradox Demo")
5. Copy the key (starts with `sk-proj-...`)
6. **Save it securely** - you won't see it again!

**Billing**:
- Requires payment method on file
- Pay-as-you-go pricing
- Estimated cost: $2-5 per full run

#### Anthropic API Key

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign in or create account
3. Navigate to **API Keys** section
4. Click **"Create Key"**
5. Copy the key (starts with `sk-ant-...`)

**Billing**:
- Requires payment method
- Claude 3.5 Sonnet: $3 per million tokens (input)
- Estimated cost: $1-3 per full run

#### HuggingFace API Key

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Sign in or create account
3. Click **"New token"**
4. Select **"Read"** permission (sufficient for StarCoder)
5. Copy the token (starts with `hf_...`)

**Billing**:
- Free tier available
- StarCoder via Inference API: Free for limited usage
- May require paid plan for heavy use

#### StarCoder License Agreement

**IMPORTANT**: Accept the model license before use:

1. Visit [https://huggingface.co/bigcode/starcoder](https://huggingface.co/bigcode/starcoder)
2. Click **"Agree and access repository"**
3. Accept the license terms

---

## Troubleshooting

### Common Issues

#### Issue 1: API Key Not Found

```
ValueError: OPENAI_API_KEY environment variable not set
```

**Solutions**:
- **Google Colab**: Ensure secret is added AND "Notebook access" is ON
- **Local**: Verify `.env` file exists and `load_dotenv()` is called
- Check for typos in secret names (case-sensitive!)

#### Issue 2: Quota Exceeded

```
Error: You exceeded your current quota, please check your plan and billing details
```

**Solutions**:
- Check billing settings in OpenAI/Anthropic dashboard
- Add payment method if not present
- Verify credit card hasn't expired
- Consider using GPT-3.5 instead of GPT-4 to reduce costs

#### Issue 3: Rate Limiting

```
RateLimitError: Rate limit reached for requests
```

**Solutions**:
- Add delays between agent calls:
  ```python
  import time
  time.sleep(2)  # Wait 2 seconds between calls
  ```
- Upgrade to higher API tier
- Reduce number of test iterations

#### Issue 4: HuggingFace Model Not Accessible

```
HuggingFaceHub error: Model bigcode/starcoder is not accessible
```

**Solutions**:
- Accept StarCoder license agreement (see above)
- Verify HuggingFace token has "Read" permission
- Try alternative model:
  ```python
  starcoder_llm = HuggingFaceHub(
      repo_id="bigcode/starcoderbase",  # Alternative
      ...
  )
  ```

#### Issue 5: Dependency Conflicts

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solutions**:
- Use fresh virtual environment
- Update pip: `pip install --upgrade pip`
- Install with legacy resolver:
  ```bash
  pip install -r requirements.txt --use-deprecated=legacy-resolver
  ```

#### Issue 6: Slow Execution

**Optimizations**:

1. **Use Faster Models**:
   ```python
   # Replace GPT-4 with GPT-3.5
   gpt4_llm = ChatOpenAI(model="gpt-3.5-turbo", ...)
   codex_llm = ChatOpenAI(model="gpt-3.5-turbo", ...)
   ```

2. **Reduce Task Complexity**:
   - Simplify project description
   - Reduce number of test cases

3. **Enable Parallel Execution** (advanced):
   ```python
   sdlc_crew = Crew(
       ...,
       process=Process.parallel  # Instead of sequential
   )
   ```

---

## Cost Estimation

### Per-Run Costs (Approximate)

| Model | Usage | Cost |
|-------|-------|------|
| Claude 3.5 Sonnet | ~5K tokens input, 3K output | $0.25 |
| GPT-4 Turbo | ~8K tokens input, 5K output | $1.50 |
| GPT-4 (Codex) | ~6K tokens input, 4K output | $1.20 |
| GPT-3.5-Turbo | ~3K tokens input, 2K output | $0.05 |
| StarCoder | API calls | Free (limited) |

**Total Estimated Cost**: $2.00 - $5.00 per full run

### Cost Reduction Strategies

1. **Use GPT-3.5 for all agents**:
   - Reduces cost to ~$0.20 per run
   - May reduce demonstration accuracy

2. **Limit iterations**:
   - Run isolated tests on subset of agents
   - Skip visualizations during testing

3. **Cache results**:
   ```python
   # Save results to avoid re-running
   with open('cached_results.json', 'w') as f:
       json.dump(results, f)
   ```

---

## Security Best Practices

### ⚠️ NEVER:
- Hardcode API keys in code
- Commit `.env` files to Git
- Share API keys in screenshots/logs
- Use production keys for testing

### ✅ ALWAYS:
- Use environment variables or secrets
- Add `.env` to `.gitignore`
- Rotate keys regularly
- Monitor usage dashboards
- Set spending limits on API accounts

---

## Next Steps

After successful setup:

1. ✅ Run the full notebook (`Runtime` → `Run all`)
2. ✅ Review the Integration Paradox metrics
3. ✅ Experiment with different tasks (Section 6)
4. ✅ Try alternative agent configurations
5. ✅ Export and analyze results

---

## Support

If you encounter issues not covered here:

1. Check the [README.md](README.md) troubleshooting section
2. Review API provider status pages:
   - [OpenAI Status](https://status.openai.com/)
   - [Anthropic Status](https://status.anthropic.com/)
   - [HuggingFace Status](https://status.huggingface.co/)
3. Open an issue in the repository

---

**Last Updated**: 2025-12-26
**Version**: 1.0.0

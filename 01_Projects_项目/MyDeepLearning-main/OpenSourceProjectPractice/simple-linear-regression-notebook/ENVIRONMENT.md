# Environment

This project does not keep `.venv` in the folder long term.

Recreate the environment when needed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The original upstream dependency file was saved as `requirements-original.txt`. It pins old packages that are not suitable for Python 3.13.

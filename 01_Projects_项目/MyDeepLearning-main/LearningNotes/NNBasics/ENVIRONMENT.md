# Environment

This project does not keep `.venv` in the folder long term.

Recreate the environment when needed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the example:

```powershell
python .\very_simple_neural_network.py
```

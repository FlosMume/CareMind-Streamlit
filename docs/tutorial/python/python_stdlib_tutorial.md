# Python Standard Library Tutorial

A practical guide to essential Python libraries used in the CareMind RAG Retriever project.

---

## Table of Contents

1. [sys - System Parameters](#1-sys---system-parameters)
2. [os - Operating System Interface](#2-os---operating-system-interface)
3. [glob - File Pattern Matching](#3-glob---file-pattern-matching)
4. [contextlib - Context Management](#4-contextlib---context-management)
5. [typing - Type Hints](#5-typing---type-hints)
6. [dotenv - Environment Variables](#6-dotenv---environment-variables)
7. [Related Libraries](#7-related-libraries)

---

## 1. sys - System Parameters

### Overview
The `sys` module provides access to system-specific parameters and functions, allowing interaction with the Python runtime environment.

### Key Usage in CareMind

**Module Import Substitution** ([retriever.py:116-121](../../rag/retriever.py))

```python
import sys

# Replace stdlib sqlite3 with pysqlite3
import pysqlite3 as _py
sys.modules["sqlite3"] = _py
```

This technique allows transparent substitution of modules. After modification, all subsequent `import sqlite3` statements will get the `pysqlite3` module instead.

### Common sys Attributes

| Attribute/Function | Description | Example |
|-------------------|-------------|---------|
| `sys.modules` | Dictionary of loaded modules | `sys.modules["sqlite3"] = new_module` |
| `sys.path` | List of module search paths | `sys.path.append("/custom/path")` |
| `sys.version` | Python version string | `"3.11.2 (main, ...)"` |
| `sys.argv` | Command-line arguments | `["script.py", "arg1", "arg2"]` |
| `sys.exit()` | Exit the interpreter | `sys.exit(1)` |
| `sys.stdout/stderr/stdin` | Standard I/O streams | `sys.stdout.write("Hello")` |

### Practical Example

```python
import sys

# Check Python version
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher required")

# Add custom module path
sys.path.insert(0, "/path/to/custom/modules")

# Access command-line arguments
if len(sys.argv) > 1:
    filename = sys.argv[1]
```

---

## 2. os - Operating System Interface

### Overview
The `os` module provides a portable way to interact with the operating system for file paths, environment variables, and directory operations.

### Key Usage in CareMind

**Path Normalization** ([retriever.py:79](../../rag/retriever.py))

```python
import os

def _abs(p: str) -> str:
    """Expand ~ and make path absolute to avoid CWD/hot-reload ambiguity."""
    return os.path.abspath(os.path.expanduser(p))
```

**Environment Variables** ([retriever.py:86-95](../../rag/retriever.py))

```python
# Set defaults
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

# Read with fallback
CHROMA_PERSIST_DIR = _abs(os.getenv("CHROMA_PERSIST_DIR", "./chroma_store"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
```

### Common os Functions

#### Path Operations

```python
# Path manipulation
os.path.join("folder", "subfolder", "file.txt")  # Cross-platform path joining
os.path.dirname("/path/to/file.txt")             # "/path/to"
os.path.basename("/path/to/file.txt")            # "file.txt"
os.path.exists("/path/to/file")                  # True/False
os.path.isfile("/path")                          # Check if file
os.path.isdir("/path")                           # Check if directory

# Path expansion
os.path.expanduser("~/documents")                # Expands ~ to home directory
os.path.abspath("../relative/path")              # Convert to absolute path
```

#### Directory Operations

```python
# Create directories
os.makedirs("/path/to/nested/dirs", exist_ok=True)

# List directory contents
files = os.listdir("/path/to/directory")

# Current working directory
cwd = os.getcwd()
os.chdir("/new/directory")
```

#### Environment Variables

```python
# Get environment variable
api_key = os.getenv("API_KEY", "default_value")

# Set environment variable (current process only)
os.environ["MY_VAR"] = "value"

# Set default if not exists
os.environ.setdefault("DEBUG", "false")
```

---

## 3. glob - File Pattern Matching

### Overview
The `glob` module finds files and directories matching Unix-style pathname patterns.

### Key Usage in CareMind

**Finding ChromaDB Files** ([retriever.py:241-247](../../rag/retriever.py))

```python
import glob
import os

def _sysdb_paths(persist_dir: str) -> List[str]:
    pats = [
        os.path.join(persist_dir, "chroma-*.db"),   # modern layout
        os.path.join(persist_dir, "chroma.sqlite"), # older
        os.path.join(persist_dir, "chroma.db"),     # very old
    ]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(p))
    return [f for f in files if os.path.isfile(f)]
```

### Pattern Syntax

| Pattern | Description | Example Match |
|---------|-------------|---------------|
| `*` | Matches any characters | `*.txt` → `file.txt`, `data.txt` |
| `?` | Matches single character | `file?.py` → `file1.py`, `fileA.py` |
| `[seq]` | Matches any character in seq | `file[0-9].py` → `file0.py`, `file5.py` |
| `[!seq]` | Matches any character not in seq | `file[!0-9].py` → `fileA.py` |
| `**` | Recursive directory match | `**/*.json` (with `recursive=True`) |

### Practical Examples

```python
import glob

# Find all text files in current directory
txt_files = glob.glob("*.txt")

# Find all Python files recursively
py_files = glob.glob("**/*.py", recursive=True)

# Find numbered files
numbered = glob.glob("data[0-9][0-9].csv")
# Matches: data01.csv, data99.csv

# Find files with multiple extensions
config_files = glob.glob("config.{json,yaml,yml}")

# Case-insensitive (on case-insensitive filesystems)
all_readme = glob.glob("[Rr][Ee][Aa][Dd][Mm][Ee].*")
```

### Advanced Usage

```python
from pathlib import Path

# Using pathlib for more Pythonic approach
p = Path(".")
# Equivalent to glob.glob("*.txt")
txt_files = list(p.glob("*.txt"))
# Recursive search
all_py = list(p.rglob("*.py"))
```

---

## 4. contextlib - Context Management

### Overview
The `contextlib` module provides utilities for working with context managers (the `with` statement) and helps manage resources safely.

### Key Usage in CareMind

**Suppressing Exceptions** ([retriever.py:269, 348, 471](../../rag/retriever.py))

```python
import contextlib

# Safely close database connections
finally:
    with contextlib.suppress(Exception):
        con.close()
```

This is cleaner than:
```python
finally:
    try:
        con.close()
    except Exception:
        pass
```

### Common contextlib Utilities

#### 1. `suppress(*exceptions)`

Ignore specified exceptions in a `with` block.

```python
import contextlib
import os

# Delete file, ignore if it doesn't exist
with contextlib.suppress(FileNotFoundError):
    os.remove('temp_file.txt')

# Multiple exception types
with contextlib.suppress(KeyError, AttributeError):
    value = data[key].attribute
```

#### 2. `closing(thing)`

Automatically close resources that have a `close()` method.

```python
from contextlib import closing
import urllib.request

with closing(urllib.request.urlopen(url)) as page:
    content = page.read()
# page.close() called automatically
```

#### 3. `@contextmanager`

Create custom context managers using a generator function.

```python
from contextlib import contextmanager
import sqlite3

@contextmanager
def database_connection(db_path):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# Usage
with database_connection('app.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
```

#### 4. `ExitStack`

Manage multiple context managers dynamically.

```python
from contextlib import ExitStack

# Open multiple files dynamically
with ExitStack() as stack:
    files = [stack.enter_context(open(f)) for f in filenames]
    # All files closed automatically
```

### Real-World Example

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name):
    """Time a code block."""
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end - start:.2f} seconds")

# Usage
with timer("Database query"):
    results = execute_complex_query()
```

---

## 5. typing - Type Hints

### Overview
The `typing` module provides runtime support for type hints, enabling static type checking and better IDE autocomplete.

### Key Usage in CareMind

**Function Annotations** ([retriever.py:66](../../rag/retriever.py))

```python
from typing import Any, Dict, List, Optional, Tuple

def search_guidelines(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Retrieve Top-k guideline chunks via Chroma.
    
    Returns a list of dicts with 'id', 'content', 'meta', 'score'.
    """
    # Implementation...
    return results
```

### Common Type Hints

#### Basic Types

```python
from typing import List, Dict, Set, Tuple, Optional, Union, Any

# Built-in types
name: str = "Alice"
age: int = 30
price: float = 19.99
active: bool = True

# Collections
numbers: List[int] = [1, 2, 3]
user_data: Dict[str, Any] = {"name": "Alice", "age": 30}
unique_ids: Set[str] = {"id1", "id2"}
coordinates: Tuple[float, float] = (10.5, 20.3)

# Optional (can be None)
middle_name: Optional[str] = None
# Equivalent to: Union[str, None]

# Union (multiple possible types)
identifier: Union[int, str] = "user123"

# Any (any type allowed)
config: Dict[str, Any] = {"timeout": 30, "debug": True}
```

#### Function Type Hints

```python
from typing import Optional, List, Dict, Callable

def process_data(
    data: List[str],
    callback: Optional[Callable[[str], int]] = None
) -> Dict[str, int]:
    """Process data with optional callback."""
    result = {}
    for item in data:
        if callback:
            result[item] = callback(item)
        else:
            result[item] = len(item)
    return result
```

#### Advanced Types

```python
from typing import TypeVar, Generic, Protocol

# Type variables for generic functions
T = TypeVar('T')

def first_element(items: List[T]) -> Optional[T]:
    return items[0] if items else None

# Generic class
class Container(Generic[T]):
    def __init__(self, value: T):
        self.value = value

# Protocol (structural subtyping)
from typing import Protocol

class Closeable(Protocol):
    def close(self) -> None: ...

def cleanup(resource: Closeable) -> None:
    resource.close()
```

#### Type Aliases

```python
from typing import Dict, List, Tuple

# Create readable aliases
MetaData = Dict[str, Any]
SearchResult = Dict[str, Any]
Coordinates = Tuple[float, float]

def search(query: str) -> List[SearchResult]:
    return [{"id": "1", "score": 0.95}]
```

### Modern Python (3.10+)

```python
# Union types with |
def process(value: int | str) -> int | str:
    return value

# Optional with |
name: str | None = None
```

---

## 6. dotenv - Environment Variables

### Overview
The `python-dotenv` package loads environment variables from a `.env` file into the application, keeping configuration separate from code.

### Key Usage in CareMind

**Loading Environment Variables** ([retriever.py:67-68](../../rag/retriever.py))

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Now environment variables from .env are available
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
```

### Creating a .env File

Create a file named `.env` in your project root:

```bash
# .env file
CHROMA_PERSIST_DIR=./my_chroma_store
CHROMA_COLLECTION=guideline_chunks_v2
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
DRUG_DB_PATH=./db/drugs.sqlite
CHROMA_TELEMETRY_OFF=1

# API Keys (never commit to git!)
OPENAI_API_KEY=sk-your-secret-key-here
DATABASE_URL=postgresql://user:password@localhost/dbname
```

### Best Practices

#### 1. Add .env to .gitignore

```bash
# .gitignore
.env
.env.local
.env.*.local
```

#### 2. Provide .env.example

```bash
# .env.example (commit this to git)
CHROMA_PERSIST_DIR=./chroma_store
CHROMA_COLLECTION=guideline_chunks
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
OPENAI_API_KEY=your-api-key-here
```

#### 3. Use with Default Values

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Always provide sensible defaults
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8000"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
```

### dotenv Functions

```python
from dotenv import load_dotenv, dotenv_values, find_dotenv, set_key

# Load from default .env location
load_dotenv()

# Load from specific file
load_dotenv(".env.production")

# Don't override existing environment variables
load_dotenv(override=False)

# Get values as dictionary without setting os.environ
config = dotenv_values(".env")
print(config["DATABASE_URL"])

# Find .env file in parent directories
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Programmatically set values
set_key(".env", "NEW_VAR", "new_value")
```

### Different Environments

```python
import os
from dotenv import load_dotenv

# Load environment-specific config
env = os.getenv("ENV", "development")
dotenv_file = f".env.{env}"

load_dotenv(dotenv_file)
```

Files structure:
- `.env.development`
- `.env.staging`
- `.env.production`

---

## 7. Related Libraries

### Similar to sys

#### platform - System Identification

```python
import platform

platform.system()           # 'Linux', 'Windows', 'Darwin'
platform.python_version()   # '3.11.2'
platform.machine()          # 'x86_64', 'arm64'
platform.processor()        # Processor name
```

#### importlib - Programmatic Import

```python
import importlib

# Import module by name
mod = importlib.import_module('package.module')

# Reload module
importlib.reload(mod)

# Check if module exists
spec = importlib.util.find_spec('optional_package')
if spec is not None:
    import optional_package
```

### Similar to os

#### pathlib - Object-Oriented Paths (Modern)

```python
from pathlib import Path

# Create path objects
home = Path.home()
config = home / ".config" / "app" / "settings.json"

# Check existence
if config.exists():
    data = config.read_text()

# Iterate files
for py_file in Path(".").rglob("*.py"):
    print(py_file)

# Create directories
(Path.home() / "new_folder").mkdir(parents=True, exist_ok=True)
```

#### shutil - High-Level File Operations

```python
import shutil

# Copy files
shutil.copy("source.txt", "destination.txt")
shutil.copytree("src_dir", "dst_dir")

# Move/rename
shutil.move("old_name.txt", "new_name.txt")

# Remove directory tree
shutil.rmtree("directory_to_delete")

# Disk usage
usage = shutil.disk_usage("/")
print(f"Free: {usage.free / (1024**3):.2f} GB")
```

#### subprocess - Run External Commands

```python
import subprocess

# Run command and capture output
result = subprocess.run(
    ["ls", "-la"],
    capture_output=True,
    text=True
)
print(result.stdout)

# Check if command succeeded
if result.returncode == 0:
    print("Success!")
```

### Similar to typing

#### dataclasses - Structured Data

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchResult:
    id: str
    content: str
    score: float
    meta: dict
    source: Optional[str] = None

# Usage
result = SearchResult(
    id="doc1",
    content="Example text",
    score=0.95,
    meta={"author": "Alice"}
)
```

#### pydantic - Data Validation

```python
from pydantic import BaseModel, validator

class SearchConfig(BaseModel):
    query: str
    k: int = 4
    threshold: float = 0.5
    
    @validator('k')
    def k_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('k must be positive')
        return v

# Automatic validation
config = SearchConfig(query="test", k=10)
```

---

## Summary Comparison Table

| Library | Purpose | Key Use Case |
|---------|---------|--------------|
| **sys** | System runtime | Module substitution, argv, exit |
| **os** | OS interface | Paths, env vars, directories |
| **glob** | Pattern matching | Find files by pattern |
| **contextlib** | Context managers | Resource cleanup, suppress exceptions |
| **typing** | Type hints | Better IDE support, static checking |
| **dotenv** | Config management | Load .env files |
| **platform** | System info | OS detection, Python version |
| **pathlib** | Modern paths | Object-oriented file paths |
| **shutil** | File operations | Copy, move, delete directories |
| **subprocess** | Run commands | Execute external programs |

---

## Best Practices

### 1. Path Handling

✅ **DO:**
```python
from pathlib import Path

config_path = Path.home() / ".config" / "app" / "settings.json"
if config_path.exists():
    data = config_path.read_text()
```

❌ **DON'T:**
```python
import os
config_path = os.path.expanduser("~") + "/.config/app/settings.json"
```

### 2. Environment Variables

✅ **DO:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not set")
```

❌ **DON'T:**
```python
# Hardcoded secrets
API_KEY = "sk-1234567890abcdef"
```

### 3. Resource Management

✅ **DO:**
```python
with contextlib.suppress(FileNotFoundError):
    os.remove("temp.txt")
```

❌ **DON'T:**
```python
try:
    os.remove("temp.txt")
except FileNotFoundError:
    pass
```

### 4. Type Hints

✅ **DO:**
```python
def process(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}
```

❌ **DON'T:**
```python
def process(items):
    return {item: len(item) for item in items}
```

---

## Additional Resources

- [Python Official Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [pathlib Guide](https://realpython.com/python-pathlib/)

---

**Last Updated:** December 17, 2025  
**Project:** CareMind RAG Retriever  
**Python Version:** 3.8+

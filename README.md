# Artificial-Intelligence

## Automatic Python File Headers

The header script automatically adds a standard student information header to Python files contained within each lab's `Source` directory.

For example, a Python file will receive a header similar to:

```python
# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab01_puja0009/Source/main.py
# Date: 09-08-2026
# Description: TODO: Add a brief one-line description.
# Usage: python Lab01_puja0009/Source/main.py
# Licence: MIT Licence
```

The script recursively searches for `.py` files inside directories matching:

```text
Lab*_puja0009/Source/
```

For example:

```text
repository/
├── scripts/
│   └── add_headers.py
│
├── Lab01_puja0009/
│   └── Source/
│       ├── main.py
│       └── example.py
│
├── Lab02_puja0009/
│   └── Source/
│       └── task.py
│
└── README.md
```

The repository root is determined automatically from the location of the script.

### Usage

Run the script from the terminal using:

```bash
python3 scripts/add_headers.py --name "Hans Pujalte" --fan "puja0009"
```

The `--name` and `--fan` arguments are required.

```text
--name      Student name to include in the header
--fan       Student FAN to include in the header
```

For example:

```bash
python3 scripts/add_headers.py \
    --name "Hans Pujalte" \
    --fan "puja0009"
```

### Adding a Licence

A licence can optionally be included using `--licence`:

```bash
python3 scripts/add_headers.py \
    --name "Hans Pujalte" \
    --fan "puja0009" \
    --licence "MIT Licence"
```

This adds an additional line to each header:

```python
# Licence: MIT Licence
```

If `--licence` is omitted, no licence line is added.

### Dry Run

Before modifying any files, it is recommended to perform a dry run:

```bash
python3 scripts/add_headers.py \
    --name "Hans Pujalte" \
    --fan "puja0009" \
    --dry-run
```

A dry run shows which files **would be modified** without actually changing them.

Example output:

```text
Would update: Lab01_puja0009/Source/main.py
Would update: Lab01_puja0009/Source/example.py
Would update: Lab02_puja0009/Source/task.py

3 Python file(s) would be updated.
```

Once the listed files have been checked, run the command again without `--dry-run` to apply the changes:

```bash
python3 scripts/add_headers.py \
    --name "Hans Pujalte" \
    --fan "puja0009"
```

Example output:

```text
Updated: Lab01_puja0009/Source/main.py
Updated: Lab01_puja0009/Source/example.py
Updated: Lab02_puja0009/Source/task.py

3 Python file(s) updated.
```

### Existing Headers

The script will not add another header to a Python file that already contains a student header.

For example, if the first 30 lines contain:

```python
# Student Name: Hans Pujalte
```

the file will be skipped:

```text
Already has header: Lab01_puja0009/Source/main.py
```

This prevents duplicate headers when the script is run multiple times.

### Shebang and Encoding Declarations

If a Python file begins with a shebang:

```python
#!/usr/bin/env python3
```

or a Python encoding declaration:

```python
# -*- coding: utf-8 -*-
```

the generated student header is inserted **after** these lines so that they remain valid.

For example:

```python
#!/usr/bin/env python3

# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab01_puja0009/Source/main.py
# Date: 09-08-2026
# Description: TODO: Add a brief one-line description.
# Usage: python Lab01_puja0009/Source/main.py

print("Hello world")
```

### After Running the Script

The script automatically generates the file path, date, student information, and usage command. However, the description is intentionally left as:

```python
# Description: TODO: Add a brief one-line description.
```

This should be manually replaced with a short description explaining the purpose of each Python file.

For example:

```python
# Description: Trains and evaluates a decision tree classifier using the Iris dataset.
```

### Recommended Workflow

1. Add or modify the Python files for the lab.
2. Run the header script using `--dry-run`:

```bash
python3 scripts/add_headers.py \
    --name "Hans Pujalte" \
    --fan "puja0009" \
    --dry-run
```

3. Check that the correct files are listed.
4. Run the script normally:

```bash
python3 scripts/add_headers.py \
    --name "Hans Pujalte" \
    --fan "puja0009"
```

5. Replace each `TODO` description with an appropriate description of the file.
6. Commit the completed files to the repository.

# Getting started

Here are the instruction for running the project locally. We recommend using python virtual environmet for dependencies.

Please refer to the [source code documentation](source_code_doc.md) for detailed documentation of the Python modules.

## Setting up your local Python virtual environment

Install python3-venv from your package manager.

!!! note "Install python3-venv"

    ```
    sudo apt install python3-venv
    ```

Create a virtual environment

!!! note "Create venv"

    ```
    python3 -m venv venv
    ```

Activate your virtual environment

!!! note "Activate venv"

    ```
    source venv/bin/activate
    ```

Install dependencies to the local virtual environment

!!! note "Install project dependencies"

    ```
    pip install -r requirementx.txt
    ```

## Running the application

Run with the default parameters:

```
python3 main.py
```

Show the help message:

```
python3 main.py --help
```

For more details, see the [documentation of the main module](main.md).

## Running the test suite

You can run the existing tests with the following command (requires `make`).

```
make test
```
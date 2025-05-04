
import importlib
import pkgutil
import sys
from argparse import ArgumentError
from typing import Sequence

import draccus
from pathlib import Path

# Constants for argument parsing
PATH_KEY = "path"  # Key used to identify path-related arguments
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"  # Suffix for plugin discovery arguments


def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """
    Extracts command-line arguments that correspond to a specific nested attribute level.

    Args:
        field_name (str): The name of the field to extract arguments for.
        args (Sequence[str] | None): The list of command-line arguments. Defaults to `sys.argv[1:]`.

    Returns:
        list[str] | None: A list of arguments corresponding to the specified field, or None if no arguments are found.

    Example:
        If the script is called with:
        python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path
        Calling get_cli_overrides("arg2") will return:
        ["--subarg1=abc", "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]  # Default to command-line arguments if none are provided

    attr_level_args = []
    detect_string = f"--{field_name}."  # Prefix to detect arguments for the specified field
    exclude_strings = (f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=", f"--{field_name}.{PATH_KEY}=")

    for arg in args:
        # Include arguments that start with the field prefix but exclude specific keys
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"  # Remove the field prefix
            attr_level_args.append(denested_arg)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    """
    Parses a specific argument from the command-line arguments.

    Args:
        arg_name (str): The name of the argument to parse.
        args (Sequence[str] | None): The list of command-line arguments. Defaults to `sys.argv[1:]`.

    Returns:
        str | None: The value of the argument if found, otherwise None.
    """
    if args is None:
        args = sys.argv[1:]  # Default to command-line arguments if none are provided

    prefix = f"--{arg_name}="  # Prefix to detect the argument
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]  # Extract the value after the prefix
    return None

def parse_int_arg(arg_name: str, args: Sequence[str] | None = None) -> int | None:
    """
    Parses the 'episode_index' argument as an integer from the command-line arguments.

    Args:
        args (Sequence[str] | None): The list of command-line arguments. Defaults to `sys.argv[1:]`.

    Returns:
        int | None: The parsed integer value of 'episode_index', or None if not found.

    Raises:
        ValueError: If 'episode_index' is not a valid integer.
    """
    if args is None:
        args = sys.argv[1:]  # Default to command-line arguments if none are provided
    arg_value = parse_arg("episode_index", args)
    
    if arg_value is not None:
        if any(char.isdigit() for char in arg_value):
            try:
                return int(arg_value)
            except ValueError:
                raise ValueError(f"{arg_name} must be a valid integer.")
        else:
            raise ValueError(f"{arg_name} must contain a digit.")
    return None

def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict:
    """
    Parses plugin-related arguments from the command-line arguments.

    Args:
        plugin_arg_suffix (str): The suffix to identify plugin-related arguments.
        args (Sequence[str]): The list of command-line arguments.

    Returns:
        dict: A dictionary of parsed plugin arguments, where keys are argument names and values are their values.

    Example:
        If the arguments are:
        ['--env.discover_packages_path=my_package', '--other_arg=value']
        Calling parse_plugin_args('discover_packages_path', args) will return:
        {'env.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for arg in args:
        if "=" in arg and plugin_arg_suffix in arg:
            key, value = arg.split("=", 1)  # Split the argument into key and value
            if key.startswith("--"):
                key = key[2:]  # Remove the leading '--' if present
            plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """Custom exception raised when a plugin fails to load."""


def load_plugin(plugin_path: str) -> None:
    """
    Loads and initializes a plugin from a specified Python package path.

    Args:
        plugin_path (str): The Python package path to the plugin (e.g., "mypackage.plugins.myplugin").

    Raises:
        PluginLoadError: If the plugin cannot be loaded due to import errors or invalid package paths.

    Notes:
        - The plugin package should handle its own registration during import.
        - All submodules in the plugin package will be imported.
    """
    try:
        # Import the main package module
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg):
        # Iterate over all submodules in the namespace package
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        # Import all submodules in the plugin package
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    """
    Retrieves the value of a path-related argument for a specific field.

    Args:
        field_name (str): The name of the field.
        args (Sequence[str] | None): The list of command-line arguments. Defaults to `sys.argv[1:]`.

    Returns:
        str | None: The value of the path argument if found, otherwise None.
    """
    return parse_arg(f"{field_name}.{PATH_KEY}", args)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    """
    Retrieves the value of a type-related argument for a specific field.

    Args:
        field_name (str): The name of the field.
        args (Sequence[str] | None): The list of command-line arguments. Defaults to `sys.argv[1:]`.

    Returns:
        str | None: The value of the type argument if found, otherwise None.
    """
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    """
    Filters out arguments related to a specific field.

    Args:
        field_to_filter (str): The name of the field to filter arguments for.
        args (Sequence[str] | None): The list of command-line arguments. Defaults to `sys.argv[1:]`.

    Returns:
        list[str]: A list of arguments with the specified field's arguments removed.
    """
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]


def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """
    Filters out path-related arguments for specific fields.

    Args:
        fields_to_filter (str | list[str]): A single field name or a list of field names.
        args (Sequence[str] | None): The list of command-line arguments. Defaults to `sys.argv[1:]`.

    Returns:
        list[str]: A list of arguments with the specified fields' path arguments removed.

    Raises:
        ArgumentError: If both a path argument and a type argument are specified for the same field.
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = args
    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [arg for arg in filtered_args if not arg.startswith(f"--{field}.")]

    return filtered_args

def convert_paths_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_str(i) for i in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj

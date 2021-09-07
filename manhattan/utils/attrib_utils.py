from typing import List, Optional, Callable
import attr


def requirements(
    validators: Optional[List] = None,
) -> List:
    """
    Return list of validators for the Attrib class. This is meant to help make
    creating the validators for attributes.

    Args:
        validators (List[validators]): a list of already initialized validators
            passed in to construct it
    """
    raise NotImplementedError("This constructor is not finished")
    if validators is None:
        validators = [attr.validators.instance_of(str)]
    return validators


def make_range_validator(min_val: float, max_val: float) -> Callable:
    """
    Return validator for range of values.

    Args:
        min_val (float): minimum value for validator
        max_val (float): maximum value for validator

    Returns:
        validator (Callable): validator for range of values
    """

    def range_validator(instance, attribute, value):
        if value < min_val or value > max_val:
            raise ValueError(
                f"Value {value} is not within range {min_val} to {max_val}"
            )

    return range_validator


def probability_validator(instance, attribute, value):
    """
    Return validator for probability.

    Args:
        value (float): value to validate

    Returns:
        None
    """
    if not isinstance(value, float):
        raise ValueError(f"{value} is not a float")
    if not 0 <= value <= 1:
        raise ValueError(f"Value {value} is not within range [0,1]")


def positive_float_validator(instance, attribute, value):
    """
    Return validator for positive float.

    Args:
        value (float): value to validate

    Returns:
        None
    """
    if not isinstance(value, float):
        raise ValueError(f"{value} is not a float")
    if value < 0:
        raise ValueError(f"Value {value} is not positive")


def positive_int_validator(instance, attribute, value) -> None:
    """
    Return validator for positive int.

    Args:
        value (int): value to validate

    Returns:
        None
    """
    if not isinstance(value, int):
        raise ValueError(f"{value} is not an int")
    if value < 0:
        raise ValueError(f"Value {value} is not positive")


def positive_int_tuple_validator(instance, attribute, value) -> None:
    """
    Return validator for positive int.

    Args:
        value (int): value to validate

    Returns:
        None
    """
    if not isinstance(value, tuple):
        raise ValueError(f"{value} is not a tuple")
    if not all(isinstance(x, int) for x in value):
        raise ValueError(f"At least one value in {value} is not an int")
    if not all(x >= 0 for x in value):
        raise ValueError(f"At least one value in {value} is negative")

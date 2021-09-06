from typing import List, Optional
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

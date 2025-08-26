import typing


class ComfyUI_SDXL_Adherence:
    """
    Example ComfyUI Custom Node

    This template provides a comprehensive starting point for building your own ComfyUI custom node.
    Edit this class to implement your node logic, define inputs/outputs, and add options.

    Features:
    - Customizable input and output types
    - Node options (with default values)
    - Category and display name
    - Example docstrings and type hints
    - Support for batch processing
    - Example error handling
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": (
                    "STRING",
                    {"default": "Hello, Comfy!", "multiline": False},
                ),
                "input_number": ("INT", {"default": 42, "min": 0, "max": 100}),
                "input_bool": ("BOOLEAN", {"default": True}),
                "input_choice": (
                    "CHOICE",
                    {
                        "choices": ["option1", "option2", "option3"],
                        "default": "option1",
                    },
                ),
            },
            "optional": {
                "input_optional": ("STRING", {"default": "Optional value"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN", "CHOICE")
    RETURN_NAMES = ("output_text", "output_number", "output_bool", "output_choice")
    FUNCTION = "run"
    CATEGORY = "custom"
    DISPLAY_NAME = "Example Custom Node"

    def run(
        self,
        input_text: str,
        input_number: int,
        input_bool: bool,
        input_choice: str,
        input_optional: typing.Optional[str] = None,
    ):
        """
        Main node logic. Processes inputs and returns outputs.

        Args:
            input_text (str): Text input from the user.
            input_number (int): Integer input.
            input_bool (bool): Boolean input.
            input_choice (str): Choice input.
            input_optional (str, optional): Optional string input.

        Returns:
            tuple: Outputs matching RETURN_TYPES.
        """
        # Example processing logic
        output_text = f"You entered: {input_text}"
        output_number = input_number * 2
        output_bool = not input_bool
        output_choice = input_choice.upper()

        # Example: use optional input if provided
        if input_optional:
            output_text += f" (Optional: {input_optional})"

        # Example error handling
        if input_number < 0:
            raise ValueError("input_number must be non-negative")

        return (output_text, output_number, output_bool, output_choice)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Optional: Implement this to control when the node is considered changed (for caching).
        """
        return True

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        """
        Optional: Implement this to validate inputs before running the node.
        Raise an exception to signal invalid input.
        """
        pass

    @classmethod
    def BATCHED(cls):
        """
        Optional: Return True if this node supports batched processing.
        """
        return False


# Starter Node
# This is a basic example node that can be used as a starting point for your custom nodes.
# You can extend this class to create more complex nodes with additional functionality.
# To use this node, save it in the appropriate directory and load it in ComfyUI.
# Make sure to test your node thoroughly and handle any edge cases.

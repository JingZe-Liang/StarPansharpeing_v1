#!/usr/bin/env python3
"""
Test cases for kwargs_to_dataclass function to verify its behavior with:
1. Fields with default_factory
2. Fields with regular default values
3. Fields without defaults
4. Nested dataclass fields
"""

import unittest
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.utilities.config_utils.to_dataclass import kwargs_to_dataclass


@dataclass
class NestedConfig:
    """A nested configuration class for testing."""

    name: str = "default_name"
    value: int = 42


@dataclass
class TestConfig:
    """Test configuration class with various field types.

    Note: In dataclasses, fields without defaults must be defined before fields with defaults.
    """

    # Fields without default value (required fields)
    host: str
    port: int

    # Optional field without default (will be set to None if not provided)
    optional_field: Optional[str] = None

    # Fields with regular default values
    timeout: int = 30
    retries: int = 3
    enabled: bool = True

    # Fields with default_factory
    items: List[str] = field(default_factory=list)
    mapping: Dict[str, int] = field(default_factory=dict)

    # Nested dataclass field
    nested: NestedConfig = field(default_factory=NestedConfig)


class TestKwargsToDataclass(unittest.TestCase):
    """Test cases for kwargs_to_dataclass function."""

    def test_default_factory_fields(self):
        """Test that fields with default_factory are properly handled.

        The kwargs_to_dataclass function should correctly initialize fields that use
        field(default_factory=...) by calling the factory function to create new instances.
        """
        # When no kwargs are provided for default_factory fields, they should use the factory
        result = kwargs_to_dataclass(TestConfig, host="localhost", port=8080)

        # Check that default_factory fields are properly initialized
        self.assertEqual(result.items, [])
        self.assertEqual(result.mapping, {})
        self.assertIsInstance(result.items, list)
        self.assertIsInstance(result.mapping, dict)

    def test_regular_default_fields(self):
        """Test that fields with regular defaults work correctly.

        Fields with regular default values should use those defaults when not provided
        in the kwargs, but should use provided values when they are given.
        """
        result = kwargs_to_dataclass(TestConfig, host="localhost", port=8080)

        # Check regular default values
        self.assertEqual(result.timeout, 30)
        self.assertEqual(result.retries, 3)
        self.assertEqual(result.enabled, True)

        # Check that we can override default values
        result_with_override = kwargs_to_dataclass(
            TestConfig,
            host="localhost",
            port=8080,
            timeout=60,
            retries=5,
            enabled=False,
        )

        self.assertEqual(result_with_override.timeout, 60)
        self.assertEqual(result_with_override.retries, 5)
        self.assertEqual(result_with_override.enabled, False)

    def test_required_fields(self):
        """Test that required fields (without defaults) must be provided.

        Fields without defaults are required and must be provided in the kwargs.
        If they are missing, the function should raise a TypeError.
        """
        # This should work - all required fields provided
        result = kwargs_to_dataclass(TestConfig, host="localhost", port=8080)
        self.assertEqual(result.host, "localhost")
        self.assertEqual(result.port, 8080)

        # This should fail - missing required field
        with self.assertRaises(TypeError):
            kwargs_to_dataclass(TestConfig, host="localhost")

        # This should fail - missing required field
        with self.assertRaises(TypeError):
            kwargs_to_dataclass(TestConfig, port=8080)

    def test_optional_fields(self):
        """Test optional fields behavior.

        Optional fields without defaults should be set to None when not provided,
        and should use provided values when given.
        """
        # Optional field not provided should be None
        result = kwargs_to_dataclass(TestConfig, host="localhost", port=8080)
        self.assertIsNone(result.optional_field)

        # Optional field provided should have the provided value
        result_with_value = kwargs_to_dataclass(TestConfig, host="localhost", port=8080, optional_field="test_value")
        self.assertEqual(result_with_value.optional_field, "test_value")

    def test_nested_dataclass_fields(self):
        """Test nested dataclass fields with default_factory.

        Nested dataclass fields that use default_factory should be properly instantiated
        with their own default values when not provided, and should accept provided instances.
        """
        # Nested field should be instantiated with defaults when not provided
        result = kwargs_to_dataclass(TestConfig, host="localhost", port=8080)
        self.assertIsInstance(result.nested, NestedConfig)
        self.assertEqual(result.nested.name, "default_name")
        self.assertEqual(result.nested.value, 42)

        # We can also provide a custom nested object
        custom_nested = NestedConfig(name="custom", value=100)
        result_with_custom = kwargs_to_dataclass(TestConfig, host="localhost", port=8080, nested=custom_nested)
        self.assertEqual(result_with_custom.nested.name, "custom")
        self.assertEqual(result_with_custom.nested.value, 100)

    def test_mixed_fields_with_values(self):
        """Test all field types together with provided values.

        When all field types are used together with provided values, the function
        should correctly assign all values to the resulting dataclass instance.
        """
        test_items = ["item1", "item2"]
        test_mapping = {"key1": 1, "key2": 2}
        custom_nested = NestedConfig(name="test", value=99)

        result = kwargs_to_dataclass(
            TestConfig,
            host="example.com",
            port=9000,
            items=test_items,
            mapping=test_mapping,
            timeout=45,
            retries=2,
            enabled=False,
            optional_field="present",
            nested=custom_nested,
        )

        # Check all values are properly set
        self.assertEqual(result.host, "example.com")
        self.assertEqual(result.port, 9000)
        self.assertEqual(result.items, test_items)
        self.assertEqual(result.mapping, test_mapping)
        self.assertEqual(result.timeout, 45)
        self.assertEqual(result.retries, 2)
        self.assertEqual(result.enabled, False)
        self.assertEqual(result.optional_field, "present")
        self.assertEqual(result.nested.name, "test")
        self.assertEqual(result.nested.value, 99)

    def test_default_factory_creates_new_instances(self):
        """Test that default_factory creates new instances for each call.

        Each call to kwargs_to_dataclass should create new instances for fields
        with default_factory, ensuring that modifications to one instance don't
        affect others.
        """
        # Create two instances without providing factory-created fields
        result1 = kwargs_to_dataclass(TestConfig, host="localhost", port=8080)
        result2 = kwargs_to_dataclass(TestConfig, host="localhost", port=8080)

        # They should have separate list and dict instances
        self.assertIsNot(result1.items, result2.items)
        self.assertIsNot(result1.mapping, result2.mapping)

        # But they should be equal in content
        self.assertEqual(result1.items, result2.items)
        self.assertEqual(result1.mapping, result2.mapping)

        # Modifying one should not affect the other
        result1.items.append("item")
        result1.mapping["key"] = 1

        self.assertEqual(len(result2.items), 0)
        self.assertEqual(len(result2.mapping), 0)
        self.assertNotIn("key", result2.mapping)

    def test_excess_kwargs_are_ignored(self):
        """Test that excess kwargs (not in the dataclass) are ignored.

        The function should only use kwargs that correspond to fields in the dataclass,
        ignoring any extra kwargs that don't match dataclass fields.
        """
        # Provide extra kwargs that don't exist in TestConfig
        result = kwargs_to_dataclass(
            TestConfig,
            host="localhost",
            port=8080,
            extra_param1="should_be_ignored",
            extra_param2=123,
            another_extra="also_ignored",
        )

        # Verify the result is still correctly created
        self.assertEqual(result.host, "localhost")
        self.assertEqual(result.port, 8080)

        # Verify default values still work
        self.assertEqual(result.timeout, 30)
        self.assertEqual(result.items, [])

        # Verify the extra parameters don't exist in the result
        self.assertFalse(hasattr(result, "extra_param1"))
        self.assertFalse(hasattr(result, "extra_param2"))
        self.assertFalse(hasattr(result, "another_extra"))


if __name__ == "__main__":
    unittest.main()

from src.schema_validation import validate_features


SCHEMA = {
    "features": [
        {"name": "Tenure Months", "type": "number", "required": True, "min": 1, "max": 72},
        {"name": "Contract", "type": "string", "required": True, "values": ["Month-to-month", "One year"]},
    ]
}


def test_accepts_values_matching_schema():
    assert validate_features({"Tenure Months": 12, "Contract": "One year"}, SCHEMA) == []


def test_rejects_missing_and_unknown_features():
    errors = validate_features({"Tenure Months": 12, "Other": "x"}, SCHEMA)
    assert any("Missing required feature: Contract" in error for error in errors)
    assert any("Unknown feature(s): Other" in error for error in errors)


def test_rejects_invalid_types_categories_and_ranges():
    errors = validate_features({"Tenure Months": 100, "Contract": "Annual"}, SCHEMA)
    assert any("Tenure Months must be between" in error for error in errors)
    assert any("Contract must be one of" in error for error in errors)


def test_rejects_boolean_as_number():
    errors = validate_features({"Tenure Months": True, "Contract": "One year"}, SCHEMA)
    assert "Feature Tenure Months must be numeric" in errors
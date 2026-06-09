def extract_features(code):
    lines = code.split("\n")

    loc = len(lines)

    num_functions = sum(
        1 for line in lines if "def " in line or "function " in line
    )

    complexity = sum(
        1 for line in lines
        if any(keyword in line for keyword in ["if", "for", "while", "switch"])
    )

    return {
        "lines_of_code": loc,
        "num_functions": num_functions,
        "complexity": complexity
    }


def classify_project(features):
    loc = features["lines_of_code"]
    complexity = features["complexity"]
    functions = features["num_functions"]

    if loc < 200 and complexity < 20 and functions < 10:
        return "simple"
    elif loc < 1000:
        return "medium"
    else:
        return "complex"
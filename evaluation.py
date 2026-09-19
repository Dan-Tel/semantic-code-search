from search import retrieve_functions

evaluation_cases = [
    {
        "query": "attach authentication token to request",
        "expected": "addAuthToken",
    },
    {
        "query": "delete credentials from request headers",
        "expected": "removeAuthToken",
    },
    {
        "query": "combine server address with API path",
        "expected": "createApiUrl",
    },
    {
        "query": "order people alphabetically",
        "expected": "sortUsersByName",
    },
    {
        "query": "exclude removed accounts",
        "expected": "filterDeletedUsers",
    },
    {
        "query": "locate account using its identifier",
        "expected": "findUserById",
    },
    {
        "query": "synchronize graph position with browser address",
        "expected": "saveViewportToUrl",
    },
    {
        "query": "keep magnification within allowed range",
        "expected": "limitZoom",
    },
]

correct = 0
for case in evaluation_cases:
    results = retrieve_functions(case["query"], top_k=1)

    function_name = results[0]["name"] if results else None

    is_correct = function_name == case["expected"]

    if is_correct:
        correct += 1

    print(f"{'PASS' if is_correct else 'FAIL'} | expected: {case["expected"]} | predicted: {function_name}")

accuracy = correct / len(evaluation_cases)

print(f"Top-1 accuracy: {accuracy:.2%}")
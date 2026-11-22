QUERIES_BY_FUNCTION = {
    "addAuthToken": [
        "Put a bearer token into an existing request's headers.",
        "Copy the request and include the supplied token as a bearer credential.",
        "Set the Authorization field to Bearer followed by the given token.",
        "Keep the other HTTP headers while attaching the supplied credential.",
        "Where is a raw token turned into a bearer authorization header?",
    ],
    "removeAuthToken": [
        "Return a copy of the request without its Authorization header.",
        "Which helper drops the bearer credential from the headers?",
        "Clear the request's authorization field while preserving its other headers.",
        "Take an HTTP request and return it without the header that carries credentials.",
        "Where is Authorization deleted from a copy of the headers?",
    ],
    "createApiUrl": [
        "Join the server's base address and a route with a slash.",
        "Turn a service root and a relative endpoint into one URL string.",
        "Which helper builds a request address from two path components?",
        "Append an endpoint to the API root to form the full address.",
        "Where does the code put a forward slash between the base and the endpoint?",
    ],
    "saveViewportToUrl": [
        "Write the map's center coordinates and zoom level into the query string.",
        "Update the address bar to reflect the current view without navigating away.",
        "Which helper replaces the current history entry with the view settings?",
        "Put the current x and y center and scale into browser URL parameters.",
        "Store the visible area settings in the address bar with replaceState.",
    ],
    "limitZoom": [
        "Cap the zoom at its upper bound and raise it if it falls below the lower bound.",
        "Which helper returns the closest allowed value when the zoom is out of range?",
        "Leave valid zoom levels alone and bring out-of-range ones back to the boundary.",
        "Apply both a floor and a ceiling to the requested magnification.",
        "Where are Math.min and Math.max used to constrain the zoom level?",
    ],
    "sortUsersByName": [
        "Return a copy of the user list ordered by each person's name.",
        "Which helper uses locale-aware comparison to order account names?",
        "Put the people in A-to-Z order without changing the original array.",
        "Order records by their name field using the language's string comparison rules.",
        "Where is localeCompare called on the names in the user list?",
    ],
    "filterDeletedUsers": [
        "Return only user records whose deletion flag is false.",
        "Which helper ignores accounts flagged as deleted?",
        "Drop soft-deleted entries from the list without deleting any records.",
        "Build a list of remaining users by checking each record's deleted field.",
        "Where is the user list screened for records that should no longer appear?",
    ],
    "findUserById": [
        "Return the first account whose ID matches the supplied value.",
        "Search the user array for the record with the requested identifier.",
        "Which helper returns undefined if no account has the given ID?",
        "Fetch one matching user from an in-memory list by comparing IDs.",
        "Where does the code scan for an exact match on the user's id field?",
    ],
}

# evaluate_cases ожидает пары (запрос, имя правильной функции).
PARAPHRASE_CASES = [
    (query, function_name)
    for function_name, queries in QUERIES_BY_FUNCTION.items()
    for query in queries
]
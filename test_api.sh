BASE="http://localhost:5050"
echo "=== GET /api/tools ==="
curl -s "$BASE/api/tools" | python3 -m json.tool
echo
echo "=== POST /api/tool (calculator) ==="
curl -s -X POST "$BASE/api/tool" \
    -H "Content-Type: application/json" \
    -d '{"name": "calculator", "arguments": "247 * 183"}' | python3 -m json.tool
echo
echo "=== POST /api/tool (sympy) ==="
curl -s -X POST "$BASE/api/tool" \
    -H "Content-Type: application/json" \
    -d '{"name": "sympy", "arguments": "{\"action\": \"solve\", \"equation\": \"2*x + 5 = 15\", \"variable\": \"x\"}"}' | python3 -m json.tool
echo
echo "=== POST /api/query (agent loop) ==="
curl -s -X POST "$BASE/api/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is 247 * 183?"}' | python3 -m json.tool
echo
echo "=== GET /api/model/info ==="
curl -s "$BASE/api/model/info" | python3 -m json.tool
echo
echo "=== POST /api/query (error: missing query) ==="
curl -s -X POST "$BASE/api/query" \
    -H "Content-Type: application/json" \
    -d '{}' | python3 -m json.tool
echo
echo "=== POST /api/tool (error: unknown tool) ==="
curl -s -X POST "$BASE/api/tool" \
    -H "Content-Type: application/json" \
    -d '{"name": "nonexistent"}' | python3 -m json.tool
echo
echo "All endpoint tests complete!"
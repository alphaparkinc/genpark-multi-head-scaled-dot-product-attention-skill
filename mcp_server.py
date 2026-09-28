import json
import sys
from client import ScaledDotProductAttention

def handle_request(req):
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "compute_attention",
                        "description": "Compute transformer scaled dot-product attention with causal mask",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "Q": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                                "K": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                                "V": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                                "causal": {"type": "boolean", "default": False}
                            },
                            "required": ["Q", "K", "V"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "compute_attention":
            out, weights = ScaledDotProductAttention.forward(args["Q"], args["K"], args["V"], args.get("causal", False))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps({"output": out, "attention_weights": weights})}]}
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()

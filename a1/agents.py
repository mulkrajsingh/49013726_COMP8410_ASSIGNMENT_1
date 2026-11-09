# Read a request and fictional catalogue; return a tool-use trace and task scores.
# The controller validates calls before executing the shared read-only tools.
# Pretrained models supply replies; this file supplies the test interface and oracle.

"""Bounded read-only retail agents and an independent, deterministic task oracle.

The catalogue is a student-authored simulation, not an IKEA feed. No tool can
access the network, execute code, change stock or place an order.
"""

import json
import math
import re

from .io import fingerprint

SYSTEM_PROMPT = """You assist a retail evaluator using a MOCK catalogue.
Use the supplied tools to establish facts; do not invent product IDs or policies.
If a product description matches several products, request clarification.
If a product does not exist, report not_found. Actions such as placing orders,
changing stock and approving discounts are unsupported. Do not perform them.
Return the final answer as JSON with exactly these keys:
{"status": "ok|not_found|needs_clarification|unsupported", "facts": {}, "message": "short explanation"}.
Include only requested facts, using these keys where relevant: product_id, name,
colour, price, currency, stock, store, return_days, eligible.
For a non-ok status use an empty facts object. Prices are numbers; stock and
return_days are integers; eligible is a boolean. Tool calls precede the final
answer. You have at most six turns and six tool calls. Do not output reasoning.
"""


# Both agents receive the same tool names and argument requirements.
def _spec(name, description, properties, required):
    """Return the shared tool definition for the supplied name and arguments."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    _spec(
        "search_products",
        "Find catalogue products by name or category.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _spec(
        "get_product",
        "Read a product's ID, name, colour, category, price and currency.",
        {"product_id": {"type": "string"}},
        ["product_id"],
    ),
    _spec(
        "get_stock",
        "Read current stock of a product at a mock store.",
        {"product_id": {"type": "string"}, "store": {"type": "string"}},
        ["product_id", "store"],
    ),
    _spec(
        "get_return_policy",
        "Read the mock return policy for a category and opened condition.",
        {"category": {"type": "string"}, "opened": {"type": "boolean"}},
        ["category", "opened"],
    ),
]


# Check the tool name and arguments before reading the mock catalogue.
# The model can only use the small set of read only tools in the experiment.
def execute_tool(catalogue, name, arguments):
    """Return a mock tool result after checking its name and arguments."""
    specifications = {}
    for tool in TOOLS:
        function = tool["function"]
        specifications[function["name"]] = function["parameters"]
    if name not in specifications:
        return {"error": "unknown_tool"}
    schema = specifications[name]
    if not isinstance(arguments, dict) or set(arguments) != set(schema["required"]):
        return {"error": "invalid_arguments"}
    for key, value in arguments.items():
        expected = schema["properties"][key]["type"]
        if expected == "string":
            if not isinstance(value, str) or not value.strip():
                return {"error": "invalid_arguments"}
        if expected == "boolean" and type(value) is not bool:
            return {"error": "invalid_arguments"}

    products = catalogue["products"]
    if name == "search_products":
        query = arguments["query"].strip().casefold()
        matches = []
        for product in products:
            searchable_text = (product["name"] + " " + product["category"]).casefold()
            if query in searchable_text:
                matches.append(
                    {
                        "product_id": product["product_id"],
                        "name": product["name"],
                        "category": product["category"],
                    }
                )
        return {"products": matches}
    if name == "get_return_policy":
        rule = catalogue["return_policies"].get(arguments["category"])
        if rule is None:
            return {"error": "unknown_category"}
        eligible = not arguments["opened"] or rule["opened_allowed"]
        return_days = 0
        if eligible:
            return_days = rule["days"]
        return {"eligible": eligible, "return_days": return_days}

    product = None
    for candidate in products:
        if candidate["product_id"] == arguments["product_id"]:
            product = candidate
            break
    if product is None:
        return {"error": "product_not_found"}
    if name == "get_product":
        details = {}
        for key, value in product.items():
            if key != "stock":
                details[key] = value
        return details
    if arguments["store"] not in catalogue["stores"]:
        return {"error": "unknown_store"}
    return {
        "product_id": product["product_id"],
        "store": arguments["store"],
        "stock": product["stock"][arguments["store"]],
    }


# Read a model reply as either tool calls or a final structured answer.
# The tool loop needs one common format even when model templates differ.
def parse_reply(raw):
    """Parse raw model text and return its reply type and contents."""
    raw = raw.strip()
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, flags=re.S)
    try:
        if blocks:
            if re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.S).strip():
                return "invalid", None
            calls = []
            for block in blocks:
                calls.append(json.loads(block))
        else:
            if raw.startswith("```json\n") and raw.endswith("```"):
                raw = raw[8:-3].strip()
            value = json.loads(raw)
            if not isinstance(value, dict):
                return "invalid", None
            if set(value) == {"tool_calls"} and isinstance(value["tool_calls"], list):
                calls = value["tool_calls"]
            elif "name" in value and "arguments" in value:
                calls = [value]
            else:
                if (
                    set(value) != {"status", "facts", "message"}
                    or value["status"]
                    not in {"ok", "not_found", "needs_clarification", "unsupported"}
                    or not isinstance(value["facts"], dict)
                    or not isinstance(value["message"], str)
                ):
                    return "invalid", None
                return "final", value
        normalized = []
        for call in calls:
            if not isinstance(call, dict):
                return "invalid", None
            call = call.get("function", call)
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                return "invalid", None
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                return "invalid", None
            normalized.append({"name": call["name"], "arguments": arguments})
        if normalized:
            return "calls", normalized
        return "invalid", None
    except (ValueError, TypeError, KeyError):
        return "invalid", None


def smol_tool_history(messages):
    """Keep earlier calls visible to SmolLM3's text-only assistant template."""
    result = []
    for message in messages:
        item = dict(message)
        if item["role"] == "assistant" and item.get("tool_calls"):
            calls = ["<tool_call>" + json.dumps(call["function"]) + "</tool_call>"
                     for call in item["tool_calls"]]
            item["content"] = "\n".join([item.get("content") or "", *calls]).strip()
        result.append(item)
    return result


# Alternate model replies with mock tool results within the shared limits.
# The trace preserves the complete request to answer workflow for later checking.
def run_agent(prompt, catalogue, complete, max_turns=6, max_tool_calls=6):
    """Return the visible conversation trace for one retail request."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    turns, calls, final = [], [], None
    termination = "turn_limit"
    for number in range(max_turns):
        response = complete(messages, TOOLS)
        # Keep the original visible reply for the later trace check.
        raw = response["text"]
        kind, value = parse_reply(raw)
        turns.append(
            {
                "turn": number + 1,
                "raw_text": raw,
                "parsed_kind": kind,
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
                "possibly_truncated": response.get("possibly_truncated", False),
            }
        )
        if response.get("possibly_truncated"):
            termination = "token_limit"
            break
        if kind == "final":
            final, termination = value, "final"
            break
        # Invalid or mixed replies stop the loop before any proposed tool runs.
        if kind != "calls":
            termination = "invalid_reply"
            break
        # Reject the whole extra group before running any of its calls.
        if len(calls) + len(value) > max_tool_calls:
            termination = "tool_limit"
            break
        formatted_calls = []
        for index, call in enumerate(value):
            formatted_calls.append(
                {
                    "id": f"call_{number + 1}_{index}",
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
            )
        messages.append(
            {"role": "assistant", "content": "", "tool_calls": formatted_calls}
        )
        for call, formatted in zip(value, formatted_calls, strict=True):
            result = execute_tool(catalogue, call["name"], call["arguments"])
            calls.append({"turn": number + 1, **call, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "name": call["name"],
                    "tool_call_id": formatted["id"],
                    "content": json.dumps(result, sort_keys=True),
                }
            )
    total_output_tokens = 0
    for turn in turns:
        total_output_tokens += turn["output_tokens"]
    return {
        "schema_version": 1,
        "catalogue_sha256": fingerprint(catalogue),
        "turns": turns,
        "tool_calls": calls,
        "final": final,
        "termination": termination,
        "limits": {"max_turns": max_turns, "max_tool_calls": max_tool_calls},
        "total_output_tokens": total_output_tokens,
    }


# Check the recorded turns, calls and final answer for consistency.
# A saved agent file must agree with its original visible model replies.
def validate_trace(result):
    """Check result and return a small summary of its saved agent trace."""
    if (
        result.get("schema_version") != 1
        or not isinstance(result.get("turns"), list)
        or not isinstance(result.get("tool_calls"), list)
        or not isinstance(result.get("catalogue_sha256"), str)
    ):
        raise ValueError("Invalid agent trace contract")
    turns, calls = result["turns"], result["tool_calls"]
    if not 1 <= len(turns) <= 6 or len(calls) > 6:
        raise ValueError("Agent trace exceeds the six-turn/six-call contract")
    if result.get("termination") not in {
        "final",
        "turn_limit",
        "tool_limit",
        "token_limit",
        "invalid_reply",
    }:
        raise ValueError("Unknown agent termination")
    turn_numbers = []
    for turn in turns:
        turn_numbers.append(turn.get("turn"))
    if turn_numbers != list(range(1, len(turns) + 1)):
        raise ValueError("Agent turns must be ordered and complete")
    for turn in turns:
        if not isinstance(turn.get("raw_text"), str):
            raise ValueError("Missing raw agent output")
        for field in ["input_tokens", "output_tokens"]:
            if type(turn.get(field)) is not int or turn[field] < 0:
                raise ValueError("Invalid agent token counts")
        if turn["output_tokens"] > 512:
            raise ValueError("Agent output exceeds the per-turn token contract")
    total_output_tokens = 0
    for turn in turns:
        total_output_tokens += turn["output_tokens"]
    if result.get("total_output_tokens") != total_output_tokens:
        raise ValueError("Agent token totals disagree")
    for call in calls:
        if (
            type(call.get("turn")) is not int
            or call["turn"] not in range(1, len(turns) + 1)
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), dict)
            or not isinstance(call.get("result"), dict)
        ):
            raise ValueError("Malformed tool trace")
    final = result.get("final")
    if result["termination"] == "final":
        kind, parsed = parse_reply(turns[-1]["raw_text"])
        if kind != "final" or parsed != final:
            raise ValueError("Final answer disagrees with the original model output")
    elif final is not None:
        raise ValueError("A non-final trace cannot claim a final answer")
    return {
        "kind": "agent",
        "turns": len(turns),
        "tool_calls": len(calls),
        "termination": result["termination"],
    }


# Replay tool calls and compare the final facts with the reviewed expected facts.
# The result is checked independently of the model that produced it.
def score_agent(result, catalogue, expected):
    """Return task and trace scores for one saved retail-agent response."""
    validate_trace(result)
    integrity = result["catalogue_sha256"] == fingerprint(catalogue)
    successful_tools = []
    errors = 0
    for call in result["tool_calls"]:
        actual = execute_tool(catalogue, call["name"], call["arguments"])
        if actual != call["result"]:
            integrity = False
        if actual.get("error") in {
            "unknown_tool",
            "invalid_arguments",
            "unknown_category",
            "unknown_store",
        }:
            errors += 1
        if actual.get("error") not in {"unknown_tool", "invalid_arguments"}:
            successful_tools.append(call["name"])

    # Check that the log contains the calls actually requested in the raw text.
    parsed_calls = []
    for turn in result["turns"]:
        kind, values = parse_reply(turn["raw_text"])
        if kind == "calls":
            for call in values:
                parsed_calls.append({"turn": turn["turn"], **call})
    recorded = []
    for call in result["tool_calls"]:
        recorded_call = {}
        for key in ["turn", "name", "arguments"]:
            recorded_call[key] = call[key]
        recorded.append(recorded_call)
    if parsed_calls != recorded:
        integrity = False

    final = result.get("final")
    passed = False
    if integrity and final and final["status"] == expected["status"]:
        if facts_equal(final["facts"], expected["facts"]):
            if set(expected.get("required_tools", [])) <= set(successful_tools):
                # Matching facts alone are insufficient if a required lookup was skipped.
                passed = True
                for required in expected.get("required_calls", []):
                    found_call = False
                    for call in result["tool_calls"]:
                        if call["name"] == required["name"] and facts_equal(
                            call["arguments"], required["arguments"]
                        ):
                            found_call = True
                            break
                    if not found_call:
                        passed = False
                        break
    return {
        "task_success": passed,
        "trace_integrity": bool(integrity),
        "tool_errors": errors,
        "agent_turns": len(result["turns"]),
        "tool_calls": len(result["tool_calls"]),
        "output_tokens": result["total_output_tokens"],
        "agent_termination": result["termination"],
    }


# Keep boolean true separate from numeric 1 when comparing facts.
def facts_equal(actual, expected):
    """Return whether actual and expected contain the same fact values."""
    if set(actual) != set(expected):
        return False
    for key, value in expected.items():
        observed = actual[key]
        if type(value) in {float, int}:
            if (
                type(observed) not in {float, int}
                or not math.isfinite(observed)
                or observed != value
            ):
                return False
        elif type(observed) is not type(value) or observed != value:
            return False
    return True

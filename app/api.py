from __future__ import annotations
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from flask import Flask, jsonify, request. render_template
from flask_cors import CORS

from threading import Lock
model_lock = Lock()

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.agent_loop import AgentLoop, AgentResult, StepRecord, make_hf_generate_fn
from tools import dispatch as tool_dispatch, list_tools, TOOL_REGISTRY
logger = logging.getLogger(__name__)

def create_app(model_path: Optional[str] = None, max_iterations: int = 5, temperature: float = 0.7, max_new_tokens: int = 1024) -> Flask:
    app = Flask(__name__)
    CORS(app)
    app.config["model_path"] = model_path
    app.config["max_iterations"] = max_iterations
    app.config["temperature"] = temperature
    app.config["max_new_tokens"] = max_new_tokens
    app.config["model_loaded"] = False
    app.config["model_name"] = "N/A"

    _state: Dict[str, Any] = {
        "generate_fn": None,
        "model": None,
        "tokenizer": None,
        "load_time_s": 0.0,
    }

    def _load_model() -> None:
        t0 = time.perf_counter()
        path = app.config["model_path"]
        logger.info("Loading model from '%s' …", path)
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=1024,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)

            _state["model"] = model
            _state["tokenizer"] = tokenizer
            _state["generate_fn"] = make_hf_generate_fn(model, tokenizer, max_new_tokens=app.config["max_new_tokens"], temperature=app.config["temperature"])
            elapsed = time.perf_counter() - t0
            _state["load_time_s"] = elapsed
            app.config["model_loaded"] = True
            app.config["model_name"] = path
            logger.info("Model loaded in %.1fs", elapsed)

        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)

    def _ok(data: Any) -> tuple:
        return jsonify({"status": "ok", "data": data, "error": None}), 200

    def _error(msg: str, status_code: int = 400) -> tuple:
        return jsonify({"status": "error", "data": None, "error": msg}), status_code

    def _step_to_dict(step: StepRecord) -> Dict[str, Any]:
        return {
            "step_index": step.step_index,
            "state": step.state.name,
            "think": step.think,
            "tool_name": step.tool_name,
            "tool_args": step.tool_args,
            "tool_result": step.tool_result,
            "final_answer": step.final_answer,
            "error": step.error,
            "elapsed_ms": round(step.elapsed_ms, 2),
        }

    def _result_to_dict(result: AgentResult) -> Dict[str, Any]:
        return {
            "answer": result.answer,
            "steps": [_step_to_dict(s) for s in result.steps],
            "total_elapsed_ms": round(result.total_elapsed_ms, 2),
            "loop_count": result.loop_count,
            "terminated_reason": result.terminated_reason,
            "full_trajectory": result.full_trajectory,
        }


    @app.route("/api/query", methods=["POST"])
    def query_agent():
        if not app.config["model_loaded"]:
            return _error("Model not loaded yet", 503)

        body = request.get_json(silent=True)
        if not body or "query" not in body:
            return _error("Missing 'query' in request body")

        user_query = body["query"].strip()
        if not user_query:
            return _error("Empty query")

        iters = body.get("max_iterations", app.config["max_iterations"])
        temp = body.get("temperature", app.config["temperature"])
        generate_fn = _state["generate_fn"]
        if temp != app.config["temperature"] and _state["model"] is not None:
            generate_fn = make_hf_generate_fn(_state["model"], _state["tokenizer"], max_new_tokens=app.config["max_new_tokens"], temperature=temp)

        agent = AgentLoop(generate_fn=generate_fn, tool_dispatch_fn=tool_dispatch, max_iterations=iters)
        try:
            with model_lock:
                result = agent.run(user_query)
            return _ok(_result_to_dict(result))
        except Exception as exc:
            logger.exception("Agent run failed")
            return _error(f"Agent execution error: {exc}", 500)

    @app.route("/api/tool", methods=["POST"])
    def execute_tool():
        body = request.get_json(silent=True)
        if not body or "name" not in body:
            return _error("Missing 'name' in request body")

        tool_name = body["name"]
        arguments = body.get("arguments", "")
        t0 = time.perf_counter()
        try:
            output = tool_dispatch(tool_name, str(arguments))
            elapsed = (time.perf_counter() - t0) * 1000
            return _ok({
                "tool": tool_name,
                "input": str(arguments),
                "output": output,
                "elapsed_ms": round(elapsed, 2),
            })
        except ValueError as exc:
            return _error(str(exc), 404)
        except Exception as exc:
            logger.exception("Tool execution failed")
            return _error(f"Tool error: {exc}", 500)

    @app.route("/api/tools", methods=["GET"])
    def get_tools():
        tools = list_tools()
        return _ok({"tools": tools, "count": len(tools)})

    @app.route("/api/health", methods=["GET"])
    def health_check():
        tool_ok = False
        try:
            probe = tool_dispatch("calculator", "1 + 1")
            tool_ok = probe.strip() == "2"
        except Exception:
            pass

        return _ok({
            "model_loaded": app.config["model_loaded"],
            "model_name": app.config["model_name"],
            "tool_pipeline": "ok" if tool_ok else "error",
            "tools_available": list_tools(),
            "load_time_s": round(_state["load_time_s"], 2),
        })

    @app.route("/api/model/info", methods=["GET"])
    def model_info():
        info: Dict[str, Any] = {
            "model_path": app.config["model_path"],
            "max_iterations": app.config["max_iterations"],
            "temperature": app.config["temperature"],
            "max_new_tokens": app.config["max_new_tokens"],
            "load_time_s": round(_state["load_time_s"], 2),
        }

        if _state["model"] is not None:
            model = _state["model"]
            info["model_type"] = type(model).__name__
            info["device"] = str(model.device) if hasattr(model, "device") else "unknown"
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                info["total_parameters"] = total_params
                info["trainable_parameters"] = trainable_params
            except Exception:
                pass

        if _state["tokenizer"] is not None:
            info["vocab_size"] = _state["tokenizer"].vocab_size

        return _ok(info)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/api", methods=["GET"])
    def api_docs():
        return _ok({
            "name": "Reasoning Agent API",
            "version": "1.0.0",
            "endpoints": {
                "POST /api/query": "Run the full ReAct agent loop",
                "POST /api/tool": "Execute a single tool directly",
                "GET  /api/tools": "List available tools",
                "GET  /api/health": "Health check",
                "GET  /api/model/info": "Model configuration",
            },
        })

    @app.errorhandler(404)
    def not_found(e):
        return _error("Endpoint not found", 404)

    @app.errorhandler(405)
    def method_not_allowed(e):
        return _error("Method not allowed", 405)

    @app.errorhandler(500)
    def internal_error(e):
        return _error("Internal server error", 500)

    with app.app_context():
        _load_model()

    return app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    app = create_app(model_path="./outputs/grpo_lora", max_iterations=5, temperature=0.7)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
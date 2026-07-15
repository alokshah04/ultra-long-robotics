# -*- coding: utf-8 -*-
"""
Lightweight test: RecordWrapper waypoint pending refresh flow.

Run (using uv):
    uv run python tests/lightweight/test_record_waypoint_pending_flow.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root


def _record_wrapper_path() -> Path:
    repo_root = find_repo_root(__file__)
    return repo_root / "src/robomme/env_record_wrapper/RecordWrapper.py"


def _load_tree() -> tuple[ast.Module, str]:
    src_path = _record_wrapper_path()
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))
    return tree, source


def _find_wrapper_class(tree: ast.Module) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RobommeRecordWrapper":
            return node
    raise AssertionError("RobommeRecordWrapper not found")


def _find_method(cls_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"Method {method_name} not found")


def _is_refresh_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
        and func.attr == "_refresh_pending_waypoint"
    )


def _is_super_step_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "step":
        return False
    value = func.value
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super"


def _is_clear_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
        and func.attr == "_clear_waypoint_caches_on_demo_end"
    )


def _assert_demo_tracking_and_clear_method(cls_node: ast.ClassDef, source: str) -> None:
    has_prev_demo_attr = "_prev_is_video_demo" in source
    assert has_prev_demo_attr, "State tracking field _prev_is_video_demo not detected"

    clear_fn = _find_method(cls_node, "_clear_waypoint_caches_on_demo_end")
    has_clear_current = False
    has_clear_pending = False
    for node in ast.walk(clear_fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    if target.attr == "_current_waypoint_action":
                        if isinstance(node.value, ast.Constant) and node.value.value is None:
                            has_clear_current = True
                    if target.attr == "_pending_waypoint":
                        if isinstance(node.value, ast.Constant) and node.value.value is None:
                            has_clear_pending = True

    assert has_clear_current, "Clean function did not empty self._current_waypoint_action"
    assert has_clear_pending, "Clean function did not empty env._pending_waypoint"


def _assert_step_refresh_before_super_step(step_fn: ast.FunctionDef) -> None:
    refresh_lines: list[int] = []
    super_step_lines: list[int] = []
    for node in ast.walk(step_fn):
        if not isinstance(node, ast.Call):
            continue
        if _is_refresh_call(node):
            refresh_lines.append(node.lineno)
        if _is_super_step_call(node):
            super_step_lines.append(node.lineno)

    assert refresh_lines, "step() did not call self._refresh_pending_waypoint"
    assert super_step_lines, "step() did not call super().step"
    assert min(refresh_lines) < min(super_step_lines), (
        "_refresh_pending_waypoint should be placed before super().step in step()"
    )


def _assert_step_demo_transition_clear_before_waypoint_write(step_fn: ast.FunctionDef) -> None:
    clear_call_lines: list[int] = []
    transition_if_lines: list[int] = []
    prev_demo_update_lines: list[int] = []
    waypoint_write_lines: list[int] = []

    for node in ast.walk(step_fn):
        if isinstance(node, ast.Call) and _is_clear_call(node):
            clear_call_lines.append(node.lineno)

        if isinstance(node, ast.If):
            src = ast.unparse(node.test)
            if "_prev_is_video_demo" in src and "not current_is_demo" in src:
                transition_if_lines.append(node.lineno)

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    if (
                        isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "_prev_is_video_demo"
                    ):
                        prev_demo_update_lines.append(node.lineno)

        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "waypoint_action":
                    waypoint_write_lines.append(key.lineno)

    assert transition_if_lines, "step() did not detect demo->non-demo boundary condition"
    assert clear_call_lines, "step() did not call _clear_waypoint_caches_on_demo_end"
    assert prev_demo_update_lines, "step() did not update self._prev_is_video_demo"
    assert waypoint_write_lines, "step() did not detect waypoint_action write"
    assert min(clear_call_lines) < min(waypoint_write_lines), (
        "Clean call must occur before writing to record_data['action']['waypoint_action']"
    )


def _assert_close_no_trailing_consume(close_fn: ast.FunctionDef) -> None:
    banned_attrs = {
        "_refresh_pending_waypoint",
        "backfill_waypoint_actions_in_buffer",
    }
    for node in ast.walk(close_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in banned_attrs:
            raise AssertionError(
                f"close() should not call {func.attr}, trailing waypoint post-processing logic detected"
            )
        if isinstance(func, ast.Name) and func.id in banned_attrs:
            raise AssertionError(
                f"close() should not call {func.id}, trailing waypoint post-processing logic detected"
            )


def main() -> None:
    print("\n[TEST] RecordWrapper waypoint pending flow")
    tree, source = _load_tree()
    cls_node = _find_wrapper_class(tree)

    _assert_demo_tracking_and_clear_method(cls_node, source)
    print("  clear ✓ demo end boundary clean function exists and clears current+pending")

    step_fn = _find_method(cls_node, "step")
    _assert_step_refresh_before_super_step(step_fn)
    print("  step ✓ _refresh_pending_waypoint is before super().step")
    _assert_step_demo_transition_clear_before_waypoint_write(step_fn)
    print("  step ✓ demo->non-demo boundary triggers clean, and precedes waypoint write")

    close_fn = _find_method(cls_node, "close")
    _assert_close_no_trailing_consume(close_fn)
    print("  close ✓ no trailing consume/backfill post-processing")

    print("\nPASS: record waypoint pending flow tests passed")


if __name__ == "__main__":
    main()

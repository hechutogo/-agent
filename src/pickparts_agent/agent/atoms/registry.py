"""Registry with hand-written JSON-schema-ish validation (no new dependency)."""
import math


_TABLE_OBJECT_ARGS = {
    "find_object": "target",
    "reach_above": "target",
    "grasp": "target",
    "carry_to": "container",
    "release_into": "container",
    "place_on": "target",
}


class AtomRegistry:
    def __init__(self, atoms=None):
        self.atoms = {}
        for atom in atoms or []:
            self.register(atom)

    def register(self, atom):
        if not getattr(atom, "name", ""):
            raise ValueError("Atom must define a name")
        self.atoms[atom.name] = atom

    def has(self, name):
        return name in self.atoms

    def get(self, name):
        return self.atoms[name]

    def validate_call(self, call):
        if not isinstance(call, dict):
            raise ValueError("Atom call must be an object")
        if set(call) - {"atom", "args"}:
            raise ValueError("Atom call accepts only atom and args")
        name = call.get("atom")
        if not isinstance(name, str) or name not in self.atoms:
            raise ValueError(f"Unknown atom: {name!r}")
        args = call.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("Atom args must be an object")
        schema = self.atoms[name].parameters or {}
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in args:
                raise ValueError(f"Missing required arg: {key}")
        for key, value in args.items():
            spec = props.get(key)
            if spec is None:
                raise ValueError(f"Unknown arg: {key}")
            expected = spec.get("type")
            if expected == "string" and not isinstance(value, str):
                raise ValueError(f"Arg {key} must be a string")
            if expected == "string" and (not value.strip() or len(value) > 160):
                raise ValueError(f"Arg {key} must be a non-empty semantic reference")
            if expected == "number" and type(value) not in (int, float):
                raise ValueError(f"Arg {key} must be a number")
            if expected == "number" and (
                    not math.isfinite(value) or value < spec.get("minimum", -math.inf)
                    or value > spec.get("maximum", math.inf)):
                raise ValueError(f"Arg {key} outside allowed range")
            if expected == "boolean" and not isinstance(value, bool):
                raise ValueError(f"Arg {key} must be a boolean")
            if "enum" in spec and value not in spec["enum"]:
                raise ValueError(f"Arg {key} must be one of {spec['enum']}")
        table_arg = _TABLE_OBJECT_ARGS.get(name)
        if table_arg is not None and args.get(table_arg) == "table":
            raise ValueError(f"{name} cannot treat table as an object")
        if name == "verify_state":
            target, at, relation = (
                args.get("target"), args.get("at"), args.get("relation"))
            if target == "table":
                raise ValueError("Table cannot be the movable verification target")
            if relation == "table" and at != "table":
                raise ValueError("Table relation needs table destination")
            if at == "table" and relation not in (None, "table"):
                raise ValueError("Table destination needs table relation")

    def catalog_for_prompt(self):
        lines = []
        for atom in self.atoms.values():
            lines.append(f"- {atom.name}: {atom.description}")
            props = (atom.parameters or {}).get("properties", {})
            if props:
                fields = []
                for key, spec in props.items():
                    enum = spec.get("enum")
                    fields.append(key + (" ∈ " + "/".join(map(str, enum))
                                        if enum else ""))
                lines.append("    args: " + ", ".join(fields))
        return "\n".join(lines)

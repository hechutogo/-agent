"""Registry with hand-written JSON-schema-ish validation (no new dependency)."""


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
            if expected == "number" and type(value) not in (int, float):
                raise ValueError(f"Arg {key} must be a number")
            if expected == "boolean" and not isinstance(value, bool):
                raise ValueError(f"Arg {key} must be a boolean")
            if "enum" in spec and value not in spec["enum"]:
                raise ValueError(f"Arg {key} must be one of {spec['enum']}")

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
